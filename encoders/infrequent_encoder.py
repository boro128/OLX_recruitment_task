import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class InfrequentEncoder(BaseEstimator, TransformerMixin):

    __slots__ = ['min_frequency', 'col_names', 'categories']

    def __init__(self, min_frequency: int, col_names: str) -> None:
        self.min_frequency = min_frequency
        self.col_names = col_names
        self.categories = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        counts = {col_name: X[col_name].value_counts() for col_name in self.col_names}
        self.categories = {col_name: counts[col_name][counts[col_name] >= self.min_frequency] for col_name in self.col_names}

        return self

    def transform(self, X: pd.DataFrame, y=None):
        X_ = X.copy()
        
        for col_name in self.col_names:
            X_[col_name] = X_[col_name].map(lambda x: x if x in self.categories[col_name] else 'Other')

        return X_
