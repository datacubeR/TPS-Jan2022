from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class HybridModel(BaseEstimator, RegressorMixin):
    def __init__(self, l_model, b_model):
        self.l_model = l_model
        self.b_model = b_model
        
    def fit(self, X, y):
        
        log_y = np.log(y)
        self.l_model.fit(X, log_y)
        linear_pred = np.exp(self.l_model.predict(X))
        y_resid = y - linear_pred
        
        self.b_model.fit(X, y_resid)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        l_predict = np.exp(self.l_model.predict(X))
        b_predict = self.b_model.predict(X)
        
        y_predict = l_predict + b_predict
        return y_predict
        
        
    