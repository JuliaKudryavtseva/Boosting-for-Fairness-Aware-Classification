import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def dfpr_score(y_true, y_pred, sensitive=None):

    if sensitive is None:
        s = np.zeros(len(y_pred)).astype(bool)
    else:
        s = sensitive
    
    wrong = y_true != y_pred
    negative = y_true == 0
    sum_neg_s = np.sum(negative[s])
    sum_neg_ns = np.sum(negative[~s])
    
    a = np.sum(wrong[s] & negative[s]) / sum_neg_s if sum_neg_s > 0 else 1
    b = np.sum(wrong[~s] & negative[~s]) / sum_neg_ns if sum_neg_ns > 0 else 1
    
    score = a - b
    
    return score
    
def dfnr_score(y_true, y_pred, sensitive=None):

    if sensitive is None:
        s = np.zeros(len(y_pred)).astype(bool)
    else:
        s = sensitive
    
    wrong = y_true != y_pred
    positive = y_true == 1
    
    sum_pos_s = np.sum(positive[s])
    sum_pos_ns = np.sum(positive[~s])
    
    a = np.sum(wrong[s] & positive[s]) / sum_pos_s if sum_pos_s > 0 else 1
    b = np.sum(wrong[~s] & positive[~s]) / sum_pos_ns if sum_pos_ns > 0 else 1
    
    score = a - b
    
    return score
    
def eq_odds_score(y_true, y_pred, sensitive):

    dfpr = dfpr_score(y_true, y_pred, sensitive)  
    dfnr = dfnr_score(y_true, y_pred, sensitive)
    
    return  abs(dfpr) + abs(dfnr)
       
def tpr_protected_score(y, y_pred, sensitive):

    y = y[sensitive == 0]
    y_pred = y_pred[sensitive == 0]
    
    tp = ((y == 1) & (y_pred == 1)).sum()
    fn = ((y == 1) & (y_pred == 0)).sum()
    
    score = tp / (tp + fn)
    
    return score

def tpr_unprotected_score(y, y_pred, sensitive):

    y = y[sensitive == 1]
    y_pred = y_pred[sensitive == 1]
    
    tp = ((y == 1) & (y_pred == 1)).sum()
    fn = ((y == 1) & (y_pred == 0)).sum()
    
    score = tp / (tp + fn)
    
    return score

def tnr_protected_score(y, y_pred, sensitive):

    y = y[sensitive == 0]
    y_pred = y_pred[sensitive == 0]
    
    tn = ((y == 0) & (y_pred == 0)).sum()
    fp = ((y == 0) & (y_pred == 1)).sum()
    
    score = tn / (tn + fp)
    
    return score

def tnr_unprotected_score(y, y_pred, sensitive):
    y = y[sensitive == 1]
    y_pred = y_pred[sensitive == 1]
    
    tn = ((y == 0) & (y_pred == 0)).sum()
    fp = ((y == 0) & (y_pred == 1)).sum()
    
    score = tn / (tn + fp)
    
    return score


def compute_metrics(y_true, y_pred, sensitive):

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    eq_odds = eq_odds_score(y_true, y_pred, sensitive)
    tnr_protected = tnr_protected_score(y_true, y_pred, sensitive)
    tnr_unprotected = tnr_unprotected_score(y_true, y_pred, sensitive)
    tpr_protected = tpr_protected_score(y_true, y_pred, sensitive)
    tpr_unprotected = tpr_unprotected_score(y_true, y_pred, sensitive)

    return accuracy, balanced_accuracy, eq_odds, tpr_protected, tpr_unprotected, tnr_protected, tnr_unprotected