import warnings 

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.utils.validation import check_X_y
from scipy.optimize import minimize

def exp_loss(a, b):
    return np.exp(b*a)

class CallbackCollector:

    def __init__(self, func, threshold=1e-3):
        self._func  = func
        self._threshold = threshold
        self._last_x = None
        self._last_val = None

    def __call__(self, x):
        self._last_x = x
        val = self._func(x)
        print('Saving intermediate value')
        if self._last_val is None:
            self._last_val = val
        elif abs(val-self._last_val) <= self._threshold:
            raise StopIteration
        else:
            self._last_val = val
        return False
    
class AdaptiveWeightsClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, base_classifier, criterion, eps=0.1):
        self.base_classifier = base_classifier
        self.criterion = criterion            
        self.eps = eps
        
    def fit(self, X, y, sensitive):
        X, y = check_X_y(X, y, accept_large_sparse=False)
        
        self.n_features_in_ = X.shape[1]
        self.classes_, y = np.unique(y, return_inverse=True)
        self._loss = exp_loss
        self.base_classifier_ = clone(self.base_classifier)
        
        def neg_criterion(params):
            self.__train(X, y, sensitive, params)
            y_pred = self.base_classifier_.predict(X)
            neg_criterion = -self.criterion(y, y_pred, sensitive)
            return neg_criterion
            
        bounds = [(0, 1), (0, 1), (0, 3), (0, 3)]
        x0 = np.array([np.random.uniform(*b) for b in bounds])

        cb = CallbackCollector(neg_criterion, threshold=1e-2)
        try:
            #Callback is required to store intermediate results since minimize not always converges
            res = minimize(neg_criterion, x0=x0, method="Powell", bounds=bounds, callback=cb)
            best_x = res.x
        except (KeyboardInterrupt, StopIteration) as e:
            best_x = cb._last_x
        finally:
            self.best_params_ = best_x
        
        return self
        
    def __train(self, X, y, sensitive, params):

        m = len(X)
        w = np.ones(m)
        w_prev = w.copy() + np.sqrt(self.eps)
        
        a_s, a_ns, b_s, b_ns = params
        
        while np.linalg.norm(w-w_prev) >= self.eps:
            
            self.base_classifier_.fit(X, y, sample_weight=w)
            
            y_pred = self.base_classifier_.predict_proba(X)[:,1]
            
            e_s = y_pred[sensitive] - y[sensitive]
            e_ns = y_pred[~sensitive] - y[~sensitive]
            
            w_prev = w.copy()
            w[sensitive] = self._loss(e_s, b_s) * a_s + self._loss(-e_s, b_s) * (1 - a_s)
            w[~sensitive] = self._loss(e_ns, b_ns) * (1 - a_ns) + self._loss(-e_ns, b_ns) * a_ns
            w = w / sum(w) * len(w)
    
    def predict(self, X):
        return self.base_classifier_.predict(X)