**Русская версия → [README_ru.md](README_ru.md)**

# Ames Housing: Price Prediction (PRO)

Reproducible pipeline for the Ames dataset. Single `sklearn.Pipeline` (imputation + OHE), leak-safe neighborhood geo-features via a custom transformer, and **Stratified K-Fold** on log-price quantiles. Includes a report, model comparison, quick RF tuning, and a Streamlit demo.

## Results
- **CV RMSE (log-target, Stratified K-Fold): `0.1400 ± 0.0102`**  
  **Folds:** `0.1290, 0.1353, 0.1316, 0.1488, 0.1552`
- **Public Kaggle LB:** **`0.14392`** (from `submission.csv`)
- **Top drivers (Permutation Importance):** `TotalSF`, `OverallQual`, `Neighborhood`, `GrLivArea`, `AgeAtSale`

## Data
- Kaggle — *House Prices: Advanced Regression Techniques* (1,460 rows, ~80 features)  
- Target: `SalePrice`, modeled as `log1p(SalePrice)`

## Method
- Preprocess: numeric → median impute; categorical → most_frequent + OHE.
- **Geo features (no leakage):** `NeighborhoodStatsTransformer` → `[Nbhd_LogPrice_Mean, Nbhd_LogPrice_Median, Nbhd_Count]` fitted only on each fold’s train split.
- Validation: 5-fold **Stratified K-Fold** on log-target quantiles.
- Models: RandomForest (baseline + tuning), optional XGBoost.

## Repro
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip -r requirements.txt
python -m src.train_pro --model rf --cv stratified
python -m src.predict
python -m scripts.make_report

