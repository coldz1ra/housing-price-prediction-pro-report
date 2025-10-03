# Ames Housing — Report

## Objective
Predict `SalePrice` from tabular features; business value: pricing and valuation support.

## Method
Single sklearn pipeline: imputation → OHE → neighborhood geo-features (no leakage) → model (RF/XGB). Target: log1p(SalePrice). Validation: 5-fold Stratified K-Fold on log-target quantiles.

## Cross-validation
CV RMSE (log-target): **0.1400** ± 0.0102

## Holdout metric
RMSE (log-target) on a 20% split: **0.0506**

## Feature importance
Top-20 permutation importance on the holdout split.

![Permutation Importance](figures/perm_importance.png)

## Segment errors
RMSE by Neighborhood (top-10 by count).

![RMSE by Neighborhood](figures/rmse_by_neighborhood.png)

## Conclusions
- Neighborhood geo-features improve stability and accuracy.
- Main price drivers: total living area, overall quality, house/remodel age.
- Log-target reduces error skew on expensive properties.

## Next steps
- Hyperparameter tuning (RF/XGB), richer geo-features.
- SHAP for local explanations.
