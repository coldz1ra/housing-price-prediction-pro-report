
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from .features import add_engineered_features, FEATURES_ALL, SAFE_NUM_FEATURES, SAFE_CAT_FEATURES
from .modeling_pro import build_preprocessor

DATA_DIR = Path('data')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    train = pd.read_csv(DATA_DIR / 'raw' / 'train.csv')
    train = add_engineered_features(train)
    cols = [c for c in FEATURES_ALL if c in train.columns]
    X = train[cols].copy()
    y_log = np.log1p(train['SalePrice'].astype(float).values)

    num = [c for c in cols if c in SAFE_NUM_FEATURES + ['TotalSF','AgeAtSale','RemodAgeAtSale','TotalBath','HasGarage','HasFireplace','RoomsPer100m2','IsRemodeled','LotAreaLog']]
    cat = [c for c in cols if c in SAFE_CAT_FEATURES]
    pre = build_preprocessor(num, cat, add_geo=True, nbhd_col='Neighborhood')

    rf = RandomForestRegressor(n_jobs=-1, random_state=42)
    pipe = Pipeline([('pre', pre), ('model', rf)])

    params = {
        'model__n_estimators': [400, 600, 800, 1000],
        'model__max_depth': [None, 8, 12, 16, 24],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 0.6, 0.8, 1.0],
    }

    rs = RandomizedSearchCV(pipe, params, n_iter=20, cv=5, scoring='neg_root_mean_squared_error',
                            n_jobs=-1, random_state=42, error_score='raise', verbose=1)
    rs.fit(X, y_log)

    best_rmse = float(-rs.best_score_)
    with open(MODELS_DIR / 'rf_tuned_metrics.json', 'w') as f:
        json.dump({'cv_rmse_mean': best_rmse, 'best_params': rs.best_params_}, f, indent=2)
    import joblib
    joblib.dump(rs.best_estimator_, MODELS_DIR / 'pipeline_rf_tuned.joblib')
    print('Saved models/pipeline_rf_tuned.joblib')

if __name__ == '__main__':
    main()
