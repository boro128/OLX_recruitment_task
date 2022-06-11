import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class OutlierEncoder(BaseEstimator, TransformerMixin):

    __slots__ = ['upper_threshold', 'lower_threshold', 'upper', 'lower']

    def __init__(self, upper_threshold=.95, lower_threshold=.05) -> None:
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        self.upper = None
        self.lower = None

    def fit(self, X, y=None):
        self.upper = np.quantile(X, self.upper_threshold, axis=0)
        self.lower = np.quantile(X, self.lower_threshold, axis=0)

        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        # X_ = pd.DataFrame(np.where(X_ > self.upper, self.upper, X_), columns=X_.columns)
        # X_ = pd.DataFrame(np.where(X_ < self.lower, self.upper, X_), columns=X_.columns)

        X_ = np.where(X_ > self.upper, self.upper, X_)
        X_ = np.where(X_ < self.lower, self.upper, X_)

        return X_
