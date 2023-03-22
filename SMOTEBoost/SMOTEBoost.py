"""
The implementation of SMOTEBoost -

a combination of the SMOTE algorithm and the boosting procedure. 
SMOTEBoost creates synthetic examples from the rare or minority 
class, thus indirectly changing the updating weights and compensating for 
skewed distributions. 

Authors: Simona Nitti, Gabriel Rozzonelli
Based on the work of the following paper:
[1] N. Chawla, A. Lazarevic, L. Hall, et K. Bowyer, « SMOTEBoost: 
    Improving Prediction  of the Minority Class in Boosting ».

"""


from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import is_classifier, clone
import numpy as np

from sklearn.tree import DecisionTreeClassifier

try:
    from .smote_sampler import SMOTE
except:
    from smote_sampler import SMOTE


class SMOTEBoostClassifier(BaseEstimator, ClassifierMixin):
    """SMOTEBoost Classifier (SMOTEBoostClassifier)
    
    Parameters
    ----------        
    base_classifier : object, default=DecisionTreeClassifier with max_depth=5
        Base classifier from which the boosted ensemble is built.
        
    n_estimators : int, default=3
        The number of base estimators.
        
    k_neighbors : int, default=5
        Number of nearest neighbors for SMOTE resampling.
        
    n_syn_samples : int, default=5
        The number of new synthetic samples on each boosting iteration.
        
    Attributes
    ----------
    classifiers_ : list
        The collection of fitted base classifiers.

    classes_ : array of shape (n_classes,)
        The classes labels.
    
    minority_class_ : int
        Class were chosen as a minority class.

    weights_ : array of shape (n_estimators,)
        The weights for each estimator in the boosted ensemble.
        
    """
    

    def __init__(self, base_classifier=DecisionTreeClassifier(max_depth=5), 
                       n_estimators=3, 
                       k_neighbors=5, 
                       n_syn_samples=5): 
        
        self.base_classifier = base_classifier  
        self.n_estimators = n_estimators  
        self.k_neighbors = k_neighbors
        self.n_syn_samples = n_syn_samples

    def fit(self, X, y):
        """Build a SMOTEBoost classifier on the training dataset (X, y).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
           The training input samples.

        y : array-like of shape (n_samples,)
           The target values (class labels) as integers.

        Returns:
        --------
        self: object
            Fitted SMOTEBoost classifier.
        """
        # Checks X and y for consistent length
        X, y = check_X_y(X, y)
                             
        # Check the binary classification problem
        check_classification_targets(y)
        
        if type_of_target(y) != "binary":
            raise ValueError("Target values should be binary")
           
        
        # Initialize lists to hold models and model weights
        self.classifiers_ = []
        self.weights_ = []

        # Find the minority class
        self.classes_, counts = np.unique(y, return_counts=True)
        self.minority_class_ = self.classes_[np.argmin(counts)]
        X_minor_samples = X[y == self.minority_class_]
        
        # Fit SMOTE on the sensitive samples       
        smote = SMOTE(k_neighbors=self.k_neighbors)
        smote.fit(X_minor_samples)

        # Distribution initialization with adjustment
        dist = np.ones_like(X) / len(y)
        dist[:, :2][self.classes_ == y[:, np.newaxis]] = 0


        for j in range(self.n_estimators):
            
            # Create new syntatical samples of the minority class
            X_synatical = smote.sample(self.n_syn_samples)
            y_synatical = np.ones(self.n_syn_samples) * self.minority_class_

            # Append to the old examples
            X_smote = np.concatenate((X, X_synatical))
            y_smote = np.concatenate((y, y_synatical))

            # Train a weak estimator on the modified distribution
            self.classifiers_.append(clone(self.base_classifier))     
            self.classifiers_[j].fit(X_smote, y_smote)

            # Compute weak hypothesis
            h = self.classifiers_[j].predict_proba(X) 

            # Compute the pseudo-loss of hypothesis and weights
            # Sum of D_t (i, y ) * (1 - h_t ( x_i , y_i ) h_t ( x_i , y ))
            h_new = np.zeros_like(h)
            h_new = h[:, :2][self.classes_ == y[:, np.newaxis]]
            epsilon = np.sum(dist[:, :2] * (1 - h_new[:,  np.newaxis] + h[:, :2]))
                    
            beta = epsilon / (1. - epsilon)
            self.weights_.append(np.log(1 / beta))

            # Update distribution where Z is a normalization constant
            z = np.sum(dist)
            exp = 0.5 * (1. + h_new[:,  np.newaxis] - h[:, :2])
            dist[:, :2] = dist[:, :2] / (z * beta**exp)
            
                        
    def predict(self, X):
        """Predict classes for input sample X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        # Checks if the estimator is fitted by verifying the presence of fitted attributes
        check_is_fitted(self)
        
        # Checks X is a non-empty 2D array containing only finite values
        X = check_array(X)
        
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        y = np.zeros(X.shape[0])

        for t in range(self.n_estimators):
            predictions += self.classifiers_[t].predict_proba(X) * self.weights_[t]
            
        for i in range(len(X)):
            try:
                y[i] = self.classes_[np.argmax(predictions[i,:])]
                
            except (ZeroDivisionError):
                raise Exception('Increase number of observaitions')
                
        
        return y
