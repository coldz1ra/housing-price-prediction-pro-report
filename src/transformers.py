
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NeighborhoodStatsTransformer(BaseEstimator, TransformerMixin):
    """Target-based stats by Neighborhood using y during fit (safe in CV).
    Creates: Nbhd_LogPrice_Mean, Nbhd_LogPrice_Median, Nbhd_Count
    """
    def __init__(self, neighborhood_col: str = 'Neighborhood'):
        self.neighborhood_col = neighborhood_col
        self.global_mean_ = None
        self.map_mean_ = None
        self.map_median_ = None
        self.map_count_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        if y is None:
            raise ValueError("NeighborhoodStatsTransformer requires y during fit")
        if self.neighborhood_col not in X.columns:
            raise ValueError(f"Column {self.neighborhood_col} not in X")
        target = np.asarray(y).reshape(-1)
        df = pd.DataFrame({'nbhd': X[self.neighborhood_col].astype(str).values, 'log_price': target})
        stats = df.groupby('nbhd')['log_price'].agg(['mean','median','count'])
        self.map_mean_ = stats['mean'].to_dict()
        self.map_median_ = stats['median'].to_dict()
        self.map_count_ = stats['count'].to_dict()
        self.global_mean_ = float(df['log_price'].mean())
        return self

    def transform(self, X: pd.DataFrame):
        nb = X[self.neighborhood_col].astype(str)
        mean = nb.map(self.map_mean_).fillna(self.global_mean_).values.reshape(-1,1)
        median = nb.map(self.map_median_).fillna(self.global_mean_).values.reshape(-1,1)
        count = nb.map(self.map_count_).fillna(0).values.reshape(-1,1)
        return np.hstack([mean, median, count])

    def get_feature_names_out(self, input_features=None):
        return np.array(['Nbhd_LogPrice_Mean','Nbhd_LogPrice_Median','Nbhd_Count'])
