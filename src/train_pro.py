
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from .features import add_engineered_features, FEATURES_ALL, SAFE_NUM_FEATURES, SAFE_CAT_FEATURES
from .modeling_pro import build_preprocessor, build_model, kfold_rmse

DATA_DIR = Path('data')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rf', choices=['rf','xgb'])
    parser.add_argument('--cv', type=str, default='stratified', choices=['kfold','stratified'])
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / 'raw' / 'train.csv')
    if 'SalePrice' not in train.columns:
        raise ValueError("Column 'SalePrice' not found. Use Kaggle House Prices dataset.")
    train['SalePrice'] = pd.to_numeric(train['SalePrice'], errors='coerce')
    train = train.dropna(subset=['SalePrice']).copy()

    train = add_engineered_features(train)
    cols = [c for c in FEATURES_ALL if c in train.columns]
    X = train[cols].copy()
    y_log = np.log1p(train['SalePrice'].astype(float).values)

    num_cols = [c for c in cols if c in SAFE_NUM_FEATURES + ['TotalSF','AgeAtSale','RemodAgeAtSale','TotalBath','HasGarage','HasFireplace','RoomsPer100m2','IsRemodeled','LotAreaLog']]
    cat_cols = [c for c in cols if c in SAFE_CAT_FEATURES]

    pre = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols, add_geo=True, nbhd_col='Neighborhood')
    model = build_model(args.model)

    pipe = Pipeline([('pre', pre), ('model', model)])

    metrics = kfold_rmse(pipe, X, y_log, n_splits=5, stratified=(args.cv=='stratified'))
    print('CV RMSE (log):', metrics['cv_rmse_mean'])

    pipe.fit(X, y_log)

    joblib.dump(pipe, MODELS_DIR / 'pipeline.joblib')
    with open(MODELS_DIR / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print('Saved to models/pipeline.joblib')

if __name__ == '__main__':
    main()
