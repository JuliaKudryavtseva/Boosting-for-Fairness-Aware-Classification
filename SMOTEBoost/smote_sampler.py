"""
The implementation of SMOTE technique.

Pick a sample randomly. Select k neighbours of observation. 
Take majority vote between the feature vector under consideration and its k 
nearest neighbors for the nominal feature value. In the case of a tie, choose 
at random. Assign that value to the new synthetic minority class sample. 
"""

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import NearestNeighbors
import numpy as np


class SMOTE:
    """SMOTE
    
    To address class imbalance performs SMOTE (ROS) sampling several times
    
    Parameters
    ----------
    k_neighbors : int, default=5
        The number of nearest neighbors.
        
    Attributes
    ----------
    k_neighbors_ : int
        The number of nearest neighbors.
    """

    def __init__(self, k_neighbors=5):
        self.k_neighbors_ = k_neighbors

    def fit(self, X):
        """Fit SMOTE on a training set, by looking for the `k_neighbors`
        nearest neighbors of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           The samples to oversample from.
        """
        self.X = check_array(X)
        self.n_features_in_ = self.X.shape[1]

        # Fit knn
        n_neighbors = self.k_neighbors_ + 1
        self.neigh = NearestNeighbors(n_neighbors=n_neighbors)
        self.neigh.fit(self.X)

        return self

    def sample(self, n_samples):
        """
        Generate synthetic observations from the training samples.

        Parameters
        ----------
        n_samples : int
            The number of new synthetic observations to generate.

        Returns
        -------
        X_new : array-like with shape (n_samples, n_features)
            The new synthetic samples.
        """
        X_new = np.zeros((n_samples, self.n_features_in_))
        
        for i in range(n_samples):
            
            # Pick a random sample
            j = np.random.randint(0, self.X.shape[0])

            # Take the k nearest neighbours 
            X_j = self.X[j].reshape(1, -1)
            neighbors = self.neigh.kneighbors(X_j, return_distance=False)
            neighbors = neighbors[:, 1:][0]
            
            # Select one of the k neighbours
            new_neigh_index = np.random.choice(neighbors)  
            
            # Measure the index between X[j] and the randomly selected neighbour
            dist = self.X[new_neigh_index] - self.X[j] 
            _adj_score = np.random.random()
            
            # Synthetize a new observation
            X_new[i] = self.X[j] + _adj_score * dist
            
      
        return X_new

