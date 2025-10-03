**English version → [README.md](README.md)**

# Housing Price Prediction (Ames) — PRO + Report

**Что внутри:**
- Стратифицированная K-Fold валидация по квантилям цены.
- Безопасные гео-фичи по `Neighborhood` (без утечек) через кастомный трансформер.
- Базовая модель RF (опционально XGBoost при наличии).
- Скрипты сравнения моделей и быстрого тюнинга RF.
- **Готовый генератор отчёта**: метрики, важность признаков, ошибки по районам, графики.

## Как запустить
```bash
# 0) Данные
#   data/raw/train.csv
#   data/raw/test.csv

# 1) Окружение
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Обучение (стратифицированный CV, RF)
python -m src.train_pro --model rf --cv stratified

# 3) Предсказания (submission.csv)
python -m src.predict

# 4) Сравнение моделей
python -m src.evaluate_models

# 5) Тюнинг RF (быстрый)
python -m src.tune_rf

# 6) Сгенерировать отчёт (графики + report.md)
python -m scripts.make_report
```

## Что появится
- `models/pipeline.joblib`, `models/metrics.json`
- `submission.csv`
- `reports/report.md` + графики в `reports/figures/`

## Результаты
- CV RMSE (лог-цена, Stratified K-Fold): 0.1400 ± 0.0102
folds: 0.1290, 0.1353, 0.1316, 0.1488, 0.1552
- Public Kaggle LB: **0.14392** (сабмит из `submission.csv`)
- Топ драйверов (perm. importance): TotalSF, OverallQual, Neighborhood, GrLivArea, AgeAtSale

'cv_rmse_mean': 0.13999415566821696
