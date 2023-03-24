from collections import Counter

import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y


from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import is_classifier, clone
import numpy as np

class RAMOSampler(object):
    """Implementation of Ranked Minority Oversampling (RAMO).

    To address class imbalance performs RAMO (ROS) sampling several times

    Parameters
    ----------
    k_neighbors_1 : int, optional (default=5)
        Number of nearest neighbors used to adjust the sampling probability of
        the minority class instances.

    k_neighbors_2 : int, optional (default=5)
        Number of nearest neighbors for synthetic data generation 
        instances.

    alpha : float, optional (default=0.25)
        Adjustment coefficient.
    """

    def __init__(
        self,
        k_neighbors_1=5,
        k_neighbors_2=5,
        alpha=0.25,
    ):
        self.k_neighbors_1 = k_neighbors_1
        self.k_neighbors_2 = k_neighbors_2
        self.alpha = alpha

    def sample(self, n_samples):
        """Generate synthetic observations from the training samples.

        Parameters
        ----------
        n_samples : int
            The number of new synthetic observations to generate.

        Returns
        -------
        X_new : array-like with shape (n_samples, n_features)
            The new synthetic samples.
        """

        X_new = np.zeros((n_samples, self.n_features))
        for i in range(n_samples):
            
            # Choose a sample 
            j = np.random.choice(self.n_minority_samples, p=self.r)

            # Find the NN for each sample 
            X_j = self.X_minority[j].reshape(1, -1)
            neighbors = self.neigh_2.kneighbors(X_j.reshape(1, -1), return_distance=False)
            neighbors = neighbors[:, 1:][0]
            neighbors_index = np.random.choice(neighbors)
            
            # Resampling
            dist = self.X_minority[neighbors_index] - self.X_minority[j]
            _adj_score = np.random.random()

            X_new[i, :] = self.X_minority[j, :] + _adj_score * dist

        return X_new

    def fit(self, X, y, sample_weight=None, minority_class_=None):
        """Train model based on input data.

        Parameters
        ----------
        X : array-like, shape = [n_total_samples, n_features]
            Holds the majority and minority samples.

        y : array-like, shape = [n_total_samples]
            Holds the class targets for samples.

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights multiplier. If None, the multiplier is 1.

        minority_class_ : int, optional (default=None)
            Minority class label.
        """
        # Define minority class
        if minority_class_ is None:
            self.classes_, counts = np.unique(y, return_counts=True)
            self.minority_class_ = self.classes_[np.argmin(counts)]
        else:
            self.minority_class_ = minority_class_

        self.X_minority = X[y == self.minority_class_]
        self.n_minority_samples, self.n_features = self.X_minority.shape
        
        # Find k nearest neighbors for to adjust the sampling probability
        neigh_1 = NearestNeighbors(n_neighbors=self.k_neighbors_1 + 1)
        neigh_1.fit(X)
        k_neighbors = neigh_1.kneighbors(self.X_minority, return_distance=False)[:, 1:]
        

        if sample_weight is None:
            sample_weight_min = np.ones(shape=(len(self.minority_class_)))
        else:
            sample_weight_min = sample_weight[y == self.minority_class_]

        # Adjust weights for the sampling probability 
        self.r = np.zeros(shape=(self.n_minority_samples))
        for i in range(self.n_minority_samples):
            
            majority_neighbors = 0
            for n in k_neighbors[i]:
                
                if y[n] != self.minority_class_:
                    majority_neighbors += 1

            self.r[i] = 1. / (1 + np.exp(-self.alpha * majority_neighbors))
            
        # Combine the weights.
        self.r = (self.r * sample_weight_min).reshape(1, -1)
        self.r = np.squeeze(normalize(self.r, axis=1, norm="l1"))

        # Learn nearest neighbors.
        self.neigh_2 = NearestNeighbors(n_neighbors=self.k_neighbors_2 + 1)
        self.neigh_2.fit(self.X_minority)

        return self
    

class RAMOBoostClassifier(AdaBoostClassifier):


    def __init__(
        self,
        n_samples=100,
        k_neighbors_1=5,
        k_neighbors_2=5,
        alpha=0.25,
        base_estimator_=DecisionTreeClassifier(),
        n_estimators=55,
        learning_rate=1.,
        random_state=0
    ):
        super().__init__()

        self.n_samples = n_samples
        self.ramo = RAMOSampler(k_neighbors_1, k_neighbors_2, alpha)
        self.base_estimator_=base_estimator_
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.alpha=alpha
        self.k_neighbors_1=k_neighbors_1
        self.k_neighbors_2=k_neighbors_2
        self.random_state=random_state

    def fit(self, X, y, sample_weight=None, minority_class_=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           The training input samples.

        y : array-like of shape (n_samples,)
           The target values (class labels) as integers.
           
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. 
            
        minority_class_ : int
        Class were chosen as a minority class.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """


        X, y = check_X_y(X, y)
        
        # Initialize weights

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0]) /  X.shape[0]
        else:
            sample_weight = sample_weight / sample_weight.sum()

        # Define minority class
        classes_, counts = np.unique(y, return_counts=True)
        self.minority_class_ = classes_[np.argmin(counts)]



        # Init boosting parameters
        self.estimators_ = []
        self.estimator_errors_ = np.ones(self.n_estimators)
        self.estimator_weights_ = np.zeros(self.n_estimators)

        random_state = check_random_state(self.random_state)
        

        for iterBoost in range(self.n_estimators):
            # RAMO sampling
            self.ramo.fit(X, y, sample_weight=sample_weight)
            X_synatical = self.ramo.sample(self.n_samples)
            y_synatical = np.ones(self.n_samples) * self.minority_class_ 
            

            # Combine the minority and majority class samples.
            X = np.concatenate((X, X_synatical))
            y = np.concatenate((y, y_synatical))

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.ones(X_synatical.shape[0]) / X.shape[0]

            # Combine the weights.
            sample_weight = np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = np.squeeze(normalize(sample_weight, axis=0, norm="l1"))


            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iterBoost,
                X, y,
                sample_weight,
                random_state,
            )
            

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iterBoost] = estimator_weight
            self.estimator_errors_[iterBoost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iterBoost < self.n_estimators - 1:
                sample_weight /= sample_weight_sum

        return self
   

