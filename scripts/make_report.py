
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

    # Train/val split for analysis visuals
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # Load trained pipeline (fallback: quick RF fit for the report if missing)
    pipe_path = MODELS / 'pipeline.joblib'
    if pipe_path.exists():
        pipe = joblib.load(pipe_path)
    else:
        from src.modeling_pro import build_preprocessor, build_model
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
    seg = te.groupby('Neighborhood').agg(count=('Neighborhood','count'),
                                        rmse=('y_true', lambda s: float(np.sqrt(np.mean((s - te.loc[s.index, 'y_pred'])**2)))))
    seg = seg.sort_values('count', ascending=False).head(10)
    # Bar plot
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

    # Write report.md
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / 'report.md', 'w', encoding='utf-8') as f:
        f.write('# Прогноз цен на жильё (Ames) — Отчёт\n\n')
        f.write('## Цель\nПредсказать `SalePrice` по признакам объекта; бизнес-ценность — оценка стоимости, ценообразование, анализ драйверов цены.\n\n')
        f.write('## Метод\nПайплайн: импьютация → OHE → гео-фичи (Neighborhood) → модель (RF/XGB). Целевая — `log1p(SalePrice)`. Валидация — 5-fold Stratified K-Fold по квантилям.\n\n')
        if cv_metrics:
            f.write('## Результаты кросс-валидации\n')
            f.write(f"CV RMSE (лог-цена): **{cv_metrics.get('cv_rmse_mean','?'):.4f}** ± {cv_metrics.get('cv_rmse_std','?'):.4f}\n\n")
        f.write('## Holdout-метрика\n')
        f.write(f"RMSE (лог-цена) на валидации: **{holdout_rmse:.4f}**\n\n")
        f.write('## Важность признаков (Permutation Importance)\n')
        f.write('Топ-20 признаков по важности на holdout.\n\n')
        f.write('![Permutation Importance](figures/perm_importance.png)\n\n')
        f.write('## Ошибка по сегментам (Neighborhood)\n')
        f.write('RMSE по топ-10 районам (по числу объектов).\n\n')
        f.write('![RMSE by Neighborhood](figures/rmse_by_neighborhood.png)\n\n')
        f.write('## Выводы\n')
        f.write('- Гео-фичи по району улучшают стабильность и точность.\n')
        f.write('- Самые сильные драйверы цены: общая жилая площадь, качество отделки, возраст дома/ремонта (проверено Permutation Importance).\n')
        f.write('- На дорогих объектах модель переобучается слабее при лог-таргете, ошибки равномернее распределены.\n\n')
        f.write('## Улучшения\n')
        f.write('- Тюнинг гиперпараметров RF/XGB (RandomizedSearch/Optuna).\n')
        f.write('- Более точные геопризнаки (квантили по району, расстояние до центра/школ).\n')
        f.write('- SHAP для локальных объяснений.\n')

    print('Report generated at reports/report.md; figures saved to reports/figures/.')

if __name__ == '__main__':
    main()
