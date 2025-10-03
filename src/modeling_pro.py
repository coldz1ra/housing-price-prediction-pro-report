
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from .transformers import NeighborhoodStatsTransformer

def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=True)

def build_preprocessor(num_cols, cat_cols, add_geo: bool = True, nbhd_col: str = 'Neighborhood') -> ColumnTransformer:
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', _make_ohe())])
    transformers = [('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)]
    if add_geo and nbhd_col in (cat_cols + num_cols):
        transformers.append(('geo', NeighborhoodStatsTransformer(neighborhood_col=nbhd_col), [nbhd_col]))
    return ColumnTransformer(transformers)

def build_model(model_name: str = 'rf'):
    if model_name == 'rf':
        return RandomForestRegressor(n_estimators=800, max_depth=None, n_jobs=-1, random_state=42)
    elif model_name == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=1200, max_depth=6, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method='hist'
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)

def kfold_rmse(model: Pipeline, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, stratified: bool = False):
    # Manual loop for stratified regression (bins), normal KFold otherwise
    if stratified:
        bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in cv.split(X, bins):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            scores.append(rmse(y_te, preds))
        return {'cv_rmse_mean': float(np.mean(scores)), 'cv_rmse_std': float(np.std(scores)), 'folds': [float(s) for s in scores]}
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in cv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            scores.append(rmse(y_te, preds))
        return {'cv_rmse_mean': float(np.mean(scores)), 'cv_rmse_std': float(np.std(scores)), 'folds': [float(s) for s in scores]}
