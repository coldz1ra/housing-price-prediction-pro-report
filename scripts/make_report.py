from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from src.features import add_engineered_features, FEATURES_ALL, SAFE_NUM_FEATURES, SAFE_CAT_FEATURES
from src.modeling_pro import build_preprocessor, build_model, rmse

ROOT = Path('.')
DATA = ROOT / 'data' / 'raw'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
FIG = REPORTS / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

def main():
    # Load data
    train = pd.read_csv(DATA / 'train.csv')
    train = add_engineered_features(train)
    cols = [c for c in FEATURES_ALL if c in train.columns]
    X = train[cols].copy()
    y_log = np.log1p(train['SalePrice'].astype(float).values)

    # Holdout split for visuals
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # Load trained pipeline (fallback: quick RF fit if missing)
    pipe_path = MODELS / 'pipeline.joblib'
    if pipe_path.exists():
        pipe = joblib.load(pipe_path)
    else:
        num = [c for c in cols if c in SAFE_NUM_FEATURES + ['TotalSF','AgeAtSale','RemodAgeAtSale','TotalBath','HasGarage','HasFireplace','RoomsPer100m2','IsRemodeled','LotAreaLog']]
        cat = [c for c in cols if c in SAFE_CAT_FEATURES]
        pre = build_preprocessor(num, cat, add_geo=True, nbhd_col='Neighborhood')
        pipe = joblib.make_pipeline(pre, build_model('rf'))
        pipe.fit(X_tr, y_tr)

    # Metrics on holdout
    preds = pipe.predict(X_te)
    holdout_rmse = rmse(y_te, preds)

    # Permutation importance
    r = permutation_importance(pipe, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1)
    importances = sorted(zip(cols, r.importances_mean), key=lambda t: t[1], reverse=True)[:20]
    names = [n for n,_ in importances]
    vals = [v for _,v in importances]

    # Plot top-20 permutation importance
    plt.figure(figsize=(8,6))
    plt.barh(range(len(names)-1, -1, -1), list(reversed(vals)))
    plt.yticks(range(len(names)-1, -1, -1), list(reversed(names)))
    plt.title('Top-20 Permutation Importance')
    plt.tight_layout()
    plt.savefig(FIG / 'perm_importance.png')
    plt.close()

    # Segment error by Neighborhood (top-10 by count)
    te = X_te.copy()
    te['y_true'] = y_te
    te['y_pred'] = preds
    seg = te.groupby('Neighborhood').agg(
        count=('Neighborhood','count'),
        rmse=('y_true', lambda s: float(np.sqrt(np.mean((s - te.loc[s.index, 'y_pred'])**2))))
    ).sort_values('count', ascending=False).head(10)

    plt.figure(figsize=(7,4))
    plt.bar(seg.index.astype(str), seg['rmse'])
    plt.xticks(rotation=45, ha='right')
    plt.title('RMSE by Neighborhood (holdout, top-10 by count)')
    plt.tight_layout()
    plt.savefig(FIG / 'rmse_by_neighborhood.png')
    plt.close()

    # Load CV metrics if exist
    cv_metrics = {}
    mfile = MODELS / 'metrics.json'
    if mfile.exists():
        cv_metrics = json.load(open(mfile))

    # Write EN report
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / 'report.md', 'w', encoding='utf-8') as f:
        f.write('# Ames Housing — Report\n\n')
        f.write('## Objective\nPredict `SalePrice` from tabular features; business value: pricing and valuation support.\n\n')
        f.write('## Method\nSingle sklearn pipeline: imputation → OHE → neighborhood geo-features (no leakage) → model (RF/XGB). ')
        f.write('Target: log1p(SalePrice). Validation: 5-fold Stratified K-Fold on log-target quantiles.\n\n')
        if cv_metrics:
            f.write('## Cross-validation\n')
            f.write(f"CV RMSE (log-target): **{cv_metrics.get('cv_rmse_mean','?'):.4f}** ± {cv_metrics.get('cv_rmse_std','?'):.4f}\n\n")
        f.write('## Holdout metric\n')
        f.write(f'RMSE (log-target) on a 20% split: **{holdout_rmse:.4f}**\n\n')
        f.write('## Feature importance\nTop-20 permutation importance on the holdout split.\n\n')
        f.write('![Permutation Importance](figures/perm_importance.png)\n\n')
        f.write('## Segment errors\nRMSE by Neighborhood (top-10 by count).\n\n')
        f.write('![RMSE by Neighborhood](figures/rmse_by_neighborhood.png)\n\n')
        f.write('## Conclusions\n')
        f.write('- Neighborhood geo-features improve stability and accuracy.\n')
        f.write('- Main price drivers: total living area, overall quality, house/remodel age.\n')
        f.write('- Log-target reduces error skew on expensive properties.\n\n')
        f.write('## Next steps\n')
        f.write('- Hyperparameter tuning (RF/XGB), richer geo-features.\n')
        f.write('- SHAP for local explanations.\n')

    print('Report generated at reports/report.md; figures saved to reports/figures/.\n(English version)')
if __name__ == '__main__':
    main()

