
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.features import add_engineered_features, FEATURES_ALL

st.set_page_config(page_title='Ames House Price Predictor', layout='centered')
st.title('🏠 Ames House Price Predictor')
st.caption('RF/XGB pipeline with geo features and stratified CV')

models_dir = Path('models')
model_file = models_dir / 'pipeline.joblib'
if not model_file.exists():
    st.warning('Model not found. Train first: `python -m src.train_pro --model rf --cv stratified`')
else:
    pipe = joblib.load(model_file)
    st.success('Model loaded')

    st.subheader('Input features (minimal demo set)')
    LotArea = st.number_input('LotArea', 100.0, 200000.0, 8450.0, 10.0)
    OverallQual = st.slider('OverallQual (1–10)', 1, 10, 7)
    YearBuilt = st.number_input('YearBuilt', 1800, 2025, 2003, 1)
    GrLivArea = st.number_input('GrLivArea', 100.0, 10000.0, 1710.0, 10.0)
    FullBath = st.slider('FullBath', 0, 5, 2)
    BedroomAbvGr = st.slider('BedroomAbvGr', 0, 10, 3)
    TotRmsAbvGrd = st.slider('TotRmsAbvGrd', 0, 20, 8)
    GarageCars = st.slider('GarageCars', 0, 5, 2)

    Neighborhood = st.text_input('Neighborhood (как в датасете)', 'CollgCr')
    YrSold = st.number_input('YrSold', 2006, 2010, 2008, 1)
    MoSold = st.slider('MoSold', 1, 12, 2)

    row = {c: np.nan for c in FEATURES_ALL}
    row.update({'LotArea': LotArea, 'OverallQual': OverallQual, 'YearBuilt': YearBuilt,
                'GrLivArea': GrLivArea, 'FullBath': FullBath, 'BedroomAbvGr': BedroomAbvGr,
                'TotRmsAbvGrd': TotRmsAbvGrd, 'GarageCars': GarageCars,
                'Neighborhood': Neighborhood, 'YrSold': YrSold, 'MoSold': MoSold})
    X = pd.DataFrame([row])
    X = add_engineered_features(X)

    if st.button('Predict'):
        log_pred = pipe.predict(X)[0]
        price = float(np.expm1(log_pred))
        st.metric('Estimated SalePrice', f'$ {price:,.0f}')
