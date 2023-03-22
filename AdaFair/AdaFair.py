"""
The implementation of AdaFair: Cumulative Fairness Adaptive Boosting
"""

import numpy as np
import pandas as pd
from sklearn.base import is_classifier, clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tqdm.notebook import tqdm
from sklearn.tree import DecisionTreeClassifier

def DFR_score (y_true, y_pred, sensitive):
    """
    DFPR_score: the difference between sensitive and non-sensitive False Positive Rates.
    DFNR_score: the difference between sensitive and non-sensitive False Negative Rates
    
    Args:
        y_true: ground truth (correct) labels.
        y_pred: predicted labels
        sensitive: array-like of shape, indicates sensitive sample. 
        
    Returns:
        DFPR_score: float (the closer to 0, the lesser is disparate mistreatment)
        DFNR_score: float (the closer to 0, the lesser is disparate mistreatment)
    """
    wrong = y_true != y_pred
    neg = y_true == 0
    pos = y_true == 1

    neg_s = np.sum(neg[sensitive])
    neg_ns = np.sum(neg[~sensitive])
    pos_s = np.sum(pos[sensitive])
    pos_ns = np.sum(pos[~sensitive])

    a1 = np.sum(wrong[sensitive] & neg[sensitive]) / neg_s if neg_s > 0 else 1
    b1 = np.sum(wrong[~sensitive] & neg[~sensitive]) / neg_ns if neg_ns > 0 else 1
    a2 = np.sum(wrong[sensitive] & pos[sensitive]) / pos_s if pos_s > 0 else 1
    b2 = np.sum(wrong[~sensitive] & pos[~sensitive]) / pos_ns if pos_ns > 0 else 1

    DFPR_score = a1 - b1
    DFNR_score = a2 - b2 

    return DFPR_score, DFNR_score

def get_fairness_cost(y_true, y_pred, y_preds, sensitive, eps):
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
        s = np.zeros(len(y_pred)).astype(bool)
    else:
        s = sensitive

    DFPR, DFNR = DFR_score (y_true, y_preds, s)

    pos_protect = ((y_true == 1) & ~s).astype(int)
    pos_unprotect = ((y_true == 1) & s).astype(int)
    neg_protect= ((y_true == -1) & ~s).astype(int)
    neg_unprotect = ((y_true == -1) & s).astype(int)
    
    if abs(DFNR) > eps:
        if DFNR > 0:
            f_cost[pos_protect & (y_true[pos_protect] != y_pred[pos_protect])] = abs(DFNR) 
        elif DFNR < 0:
            f_cost[pos_unprotect & (y_true[pos_unprotect] != y_pred[pos_unprotect])] = abs(DFNR) 
    if abs(DFPR) > eps:
        if DFPR > 0:
            f_cost[neg_protect & (y_true[neg_protect] != y_pred[neg_protect])] = abs(DFPR) 
        elif DFPR < 0:
            f_cost[neg_unprotect & (y_true[neg_unprotect] != y_pred[neg_unprotect])] = abs(DFPR) 

    return f_cost

class AdaFairClassifier(BaseEstimator, ClassifierMixin):
    """
    AdaFair Classifier

    Args:
        base_clf: object, this base estimator is used to build a boosted ensemble, which supports for sample weighting.
        n_ests: int, number of base estimators.
        epsilon [default=1e-4]: float, the error threshold.
        c [default=1]: float, the balancing coefficient for number of base classifier optimizer.
        fairness_cost [default=None]: function, is used to predict.

    Attributes:
        n_features: int, the number of features that is fitted by the classifier.
        opt: int, the optimal number of base estimators.
        list_alpha: list, includes the weights of base estimators.
        list_clfs: list, includes the base estimators.
        labels : ndarray of shape (n_classes,), the classes labels.
        
    """
    def __init__(self, base_clf, n_ests, c = 1, epsilon = 1e-4, fairness_cost = None):
        self.base_clf = base_clf 
        self.n_ests = n_ests 
        self.c = c 
        self.epsilon = epsilon
        self.fairness_cost = fairness_cost

    def fit(self, X, y, sensitive = None):    
        self.n_features = X.shape[1]
        self.opt = 1
        self.list_alpha = []
        self.list_clfs = []
        self.labels, y = np.unique(y, return_inverse=True)
        y_af = (2 * y - 1).astype(int)     

        if sensitive is None:
            sensitive = np.zeros(len(y)).astype(bool)

        n_samples = X.shape[0]
        distribution = np.ones(n_samples, dtype=float) / n_samples
        min_error = np.inf
        y_preds = 0
            
        for i in range(self.n_ests):
    
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
            if self.fairness_cost is None:
                self.fairness_cost = get_fairness_cost
            eps = self.epsilon
            f_cost = self.fairness_cost(y_af, y_pred, y_preds, sensitive, eps)
            
            # Update weights of instances
            distribution = 1/1.*distribution*np.exp(alpha*cfd*(y_af!=y_pred))*(1+f_cost)

            # Get the sign of the weighted predictions
            sign_y = np.sign(y_preds)
            sign_y = (1 + sign_y) / 2
            
            TP = ((y == 1) & (sign_y == 1)).sum()
            TN = ((y == 0) & (sign_y == 0)).sum()
            FP = ((y == 0) & (sign_y == 1)).sum()
            FN = ((y == 1) & (sign_y == 0)).sum()

            # Balanced Error Rate score
            BER = 1 - (TP / (TP + FN) + TN / (TN + FP)) / 2

            # Error Rate score
            ER = (FN + FP) / (TP + TN + FN + FP)

            # Equalized Odds classification score
            DFPR, DFNR = DFR_score (y, sign_y, sensitive)
            EO = abs(DFPR) + abs(DFNR)
            
            c = self.c
            error = c * BER + (1-c) * ER + EO
            if min_error > error:
            # The minimum of the sum of BER, ER and Eq.Odds scores
                min_error = error
            # The optimal number of base classifiers
                self.opt = i + 1
                
        return self

    def predict(self, X, end="opt"):
        
        if end == "opt":
            end = self.opt

        final_pred = np.zeros(X.shape[0])

        for alpha, clf in zip(self.list_alpha[:end], self.list_clfs[:end]):
            final_pred += alpha * clf.predict(X)

        final = ((1 + np.sign(final_pred)) / 2).astype(int)

        return self.labels[final]
