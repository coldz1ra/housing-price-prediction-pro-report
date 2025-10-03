
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from .features import add_engineered_features, FEATURES_ALL, SAFE_NUM_FEATURES, SAFE_CAT_FEATURES
from .modeling_pro import build_preprocessor, build_model, kfold_rmse

DATA_DIR = Path('data')

def main():
    train = pd.read_csv(DATA_DIR / 'raw' / 'train.csv')
    train = add_engineered_features(train)
    cols = [c for c in FEATURES_ALL if c in train.columns]
    X = train[cols].copy()
    y_log = np.log1p(train['SalePrice'].astype(float).values)

    num = [c for c in cols if c in SAFE_NUM_FEATURES + ['TotalSF','AgeAtSale','RemodAgeAtSale','TotalBath','HasGarage','HasFireplace','RoomsPer100m2','IsRemodeled','LotAreaLog']]
    cat = [c for c in cols if c in SAFE_CAT_FEATURES]
    pre = build_preprocessor(num, cat, add_geo=True, nbhd_col='Neighborhood')

    results = {}
    pipe_rf = Pipeline([('pre', pre), ('model', build_model('rf'))])
    m_rf = kfold_rmse(pipe_rf, X, y_log, stratified=True)
    results['RandomForest'] = m_rf

    try:
        pipe_xgb = Pipeline([('pre', pre), ('model', build_model('xgb'))])
        m_xgb = kfold_rmse(pipe_xgb, X, y_log, stratified=True)
        results['XGBoost'] = m_xgb
    except Exception as e:
        results['XGBoost'] = {'error': str(e)}

    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
