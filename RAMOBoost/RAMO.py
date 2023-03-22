from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class RankedMinorityOversampler:
    """This class implements the Ranked Minority Oversampling (RAMO) technique.
    It oversamples the minority class by selecting samples based on a specified
    sampling distribution.
    """
    
    def __init__(self, k_neighbors_1=5, k_neighbors_2=5, alpha=0.3, random_state=None):
        """
        Constructor for the RankedMinorityOversampler class.

        Parameters
        ----------
        k_neighbors_1 : int, optional (default=5)
            Number of nearest neighbors to consider when adjusting the sampling probability
            for minority examples.
        k_neighbors_2 : int, optional (default=5)
            Number of nearest neighbors to consider when generating synthetic data instances.
        alpha : float, optional (default=0.3)
            Scaling coefficient used for generating synthetic data instances.
        random_state : int or None, optional (default=None)
            Seed used by the random number generator. If None, the generator is initialized
            using the RandomState instance of numpy.
        """
        self.k_neighbors_adjustment = k_neighbors_1
        self.k_neighbors_synthetic_data = k_neighbors_2
        self.alpha = alpha
        self.random_state = random_state

    
    def generate_synthetic_samples(self, n_samples):
        """Generate synthetic samples.
        
        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.
        
        Returns
        -------
        synthetic_samples : array, shape = [n_samples, n_features]
            Synthetic samples.
        """
        np.random.seed(seed=self.random_state)
        
        synthetic_samples = np.zeros(shape=(n_samples, self.n_features))
        for i in range(n_samples):
            # Choose a sample according to the sampling distribution, r.
            j = np.random.choice(self.n_minority_samples, p=self.sampling_distribution)
            
            # Find the nearest neighbors for each sample.
            nearest_neighbors = self.nearest_neighbors_synthetic_data.kneighbors(
                self.minority_samples[j].reshape(1, -1), return_distance=False
            )[:, 1:]
            nn_index = np.random.choice(nearest_neighbors[0])
            
            dif = self.minority_samples[nn_index] - self.minority_samples[j]
            gap = np.random.random()
            
            synthetic_samples[i, :] = self.minority_samples[j, :] + gap * dif[:]

        return synthetic_samples
    
     def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ("SAMME", "SAMME.R"):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator_ is None or isinstance(
            self.base_estimator_, (BaseDecisionTree, BaseForest)
        )):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = "csc"
        else:
            raise ValueError('Wrong base_estimator')

        X, y = check_X_y(
            X,
            y,
            accept_sparse=accept_sparse,
            dtype=dtype,
            y_numeric=is_regressor(self),
        )

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples."
                )

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

#         # Check parameters.
#         self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)
        

        for iboost in range(self.n_estimators):
            # RAMO step.
            self.ramo.fit(X, y, sample_weight=sample_weight)
            X_syn = self.ramo.sample(self.n_samples)
            y_syn = np.full(
                X_syn.shape[0], fill_value=self.minority_target, dtype=np.int64
            )
            

            # Combine the minority and majority class samples.
            X = np.vstack((X, X_syn))
            y = np.append(y, y_syn)

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # Combine the weights.
            sample_weight = np.append(
                sample_weight, sample_weight_syn
            ).reshape(-1, 1)
            sample_weight = np.squeeze(
                normalize(sample_weight, axis=0, norm="l1")
            )


            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X,
                y,
                sample_weight,
                random_state,
            )
            

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self
  

