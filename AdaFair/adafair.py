"""
The implementation of AdaFair.
"""
# Based on the work of the following paper: 
# [1] V. Iosifidis, et E. Ntoutsi, « AdaFair: Cumulative Fairness
#     Adaptive Boosting ».

import numpy as np
import pandas as pd
from sklearn.base import is_classifier, clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from .metrics import ber_score, er_score, eq_odds_score

def fairness_cost(y_true, y_pred, y_preds, sensitive, eps):
    """
    Compute the fairness cost for sensitive features.
    
    Args:
        y_true: 1-D array, the true target values.
        y_pred: 1-D array, the predicted values.
        y_preds: 1-D array, the cumulative prediction values.
        sensitive: array-like of shape, indicates sensitive sample. 
        eps: float, the error threshold.
    
    Returns:
        f_cost: 1-D array, the fairness cost for each instance.
    """
    f_cost = np.zeros((len(y_true),))

    if sensitive is None:
        s = np.zeros(len(y_preds)).astype(bool)
    else:
        s = sensitive

    wrong = y_true != y_preds
    neg = y_true == 0
    pos = y_true == 1

    neg_s = np.sum(neg[s])
    neg_ns = np.sum(neg[~s])
    pos_s = np.sum(pos[s])
    pos_ns = np.sum(pos[~s])

    a1 = np.sum(wrong[s] & neg[s]) / neg_s if neg_s > 0 else 1
    b1 = np.sum(wrong[~s] & neg[~s]) / neg_ns if neg_ns > 0 else 1
    a2 = np.sum(wrong[s] & pos[s]) / pos_s if pos_s > 0 else 1
    b2 = np.sum(wrong[~s] & pos[~s]) / pos_ns if pos_ns > 0 else 1

    dfpr = a1 - b1
    dfnr = a2 - b2 
    
    pos_protect = ((y_true == 1) & ~s).astype(int)
    pos_unprotect = ((y_true == 1) & s).astype(int)
    neg_protect= ((y_true == -1) & ~s).astype(int)
    neg_unprotect = ((y_true == -1) & s).astype(int)
    
    if abs(dfnr) > eps:
        if dfnr > 0:
            f_cost[pos_protect & (y_true[pos_protect] != y_pred[pos_protect])] = abs(dfnr) 
        elif dfnr < 0:
            f_cost[pos_unprotect & (y_true[pos_unprotect] != y_pred[pos_unprotect])] = abs(dfnr) 
    if abs(dfpr) > eps:
        if dfpr > 0:
            f_cost[neg_protect & (y_true[neg_protect] != y_pred[neg_protect])] = abs(dfpr) 
        elif dfpr < 0:
            f_cost[neg_unprotect & (y_true[neg_unprotect] != y_pred[neg_unprotect])] = abs(dfpr) 

    return f_cost

class AdaFair(BaseEstimator, ClassifierMixin):
    """
    AdaFair Classifier

    Args:
        base_clf: object, this base estimator is used to build a boosted ensemble, which supports for sample weighting.
        n_ests: int, number of base estimators.
        epsilon [default=1e-3]: float, the error threshold.
        c: float, the balancing coefficient for number of base classifier optimizer.
        fairness_cost [default=None]: function, is used to predict.

    Attributes:
        n_features: int, the number of features that is fitted by the classifier.
        opt: int, the optimal number of base estimators.
        list_alpha: list, includes the weights of base estimators.
        list_clfs: list, includes the base estimators.
        labels : ndarray of shape (n_classes,), the classes labels.
        
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_clf = kwargs.get("base_clf", base_clf = DecisionTreeClassifier(max_depth=1))
        self.n_ests = kwargs.get("n_ests", n_ests = 100)
        self.c = kwargs.get("c", c = 1)
        self.epsilon = kwargs.get("fairness_cost", epsilon = 1e-3)
        self.fairness_cost = kwargs.get("fairness_cost", fairness_cost = None)

    def fit(self, X, y, sensitive = None):    
        self.n_features = X.shape[1]
        self.opt = 1
        self.list_alpha = []
        self.list_clfs = []
        self.labels, y = np.unique(y, return_inverse=True)
        y_af = (2 * y - 1).astype(int)     
        
        n_samples = X.shape[0]
        distribution = np.ones(n_samples, dtype=float) / n_samples
        min_error = np.inf
        y_preds = 0
            
        for i in range(self.n_estimators):
    
            # Train the base estimator
            self.list_clfs.append(clone(self.base_clf))     
            self.list_clfs[-1].fit(X, y_af, sample_weight=distribution)
            
            # Get predictions and prediction probabilities
            y_pred = self.list_clfs[-1].predict(X)
            prob = self.list_clfs[-1].predict_proba(X)[:,1]

            # Compute the confidence score derived from prediction probabilities
            cfd = abs(prob/0.5 - 1)

            # Compute the weight for a current base estimator
            n = ((y_af != y_pred) * distribution).sum() / distribution.sum()
            alpha = np.log((1-n)/n) / 2
            self.list_alpha.append(alpha)
            
            # Update of weighted votes of all fitted base estimators
            y_preds += y_pred * alpha
            
            # Compute the fairness cost for the current base learner predictions
            eps = self.epsilon
            f_cost = self.fairness_cost(y_af, y_pred, y_preds, eps)
            
            # Update weights of instances
            distribution = 1/1.*distribution*np.exp(alpha*cfd*(y_af!=y_pred))*(1+f_cost)

            # Get the sign of the weighted predictions
            y_preds_s = np.sign(y_preds)
            y_preds_s = (1 + y_preds_s) / 2
            
            # Find the optimal number of base classifiers, as the minimum of the sum of BER, ER and Eq.Odds scores
            c = self.c
            error = c * ber_score(y, y_preds_s) + (1-c) * er_score(y, y_preds_s) + eq_odds_score(y, y_preds_s, sensitive)
            
            if min_error > error:
                min_error = error
                self.opt = i + 1
                
        return self
    def predict(self, X, end="optimum"):
        
        if end == "optimum":
            end = self.opt

        final_pred = np.zeros(X.shape[0])

        for alpha, clf in zip(self.list_alpha[:end], self.list_clfs[:end]):
            final_pred += alpha * clf.predict(X)
        
        out = np.sign(final_pred)
        out = ((1 + out) / 2).astype(int)

        return self.labels[out]
