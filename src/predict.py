
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from .features import add_engineered_features, FEATURES_ALL

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

def main():
    pipe = joblib.load(MODELS_DIR / 'pipeline.joblib')
    test = pd.read_csv(DATA_DIR / 'raw' / 'test.csv')
    test = add_engineered_features(test)
    cols = [c for c in FEATURES_ALL if c in test.columns]
    X_test = test[cols].copy()

    log_preds = pipe.predict(X_test)
    preds = np.expm1(log_preds)
    pd.DataFrame({'Id': test['Id'], 'SalePrice': preds}).to_csv('submission.csv', index=False)
    print('Saved submission.csv')

if __name__ == '__main__':
    main()
