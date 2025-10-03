
from __future__ import annotations
import pandas as pd
import numpy as np

SAFE_NUM_FEATURES = [
    'MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd',
    'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF',
    '2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath',
    'HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces',
    'GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
    '3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold'
]

SAFE_CAT_FEATURES = [
    'MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig',
    'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
    'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',
    'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
    'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
    'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
    'PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'
]

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['TotalSF'] = out.get('TotalBsmtSF', 0) + out.get('1stFlrSF', 0) + out.get('2ndFlrSF', 0)
    if 'YrSold' in out and 'YearBuilt' in out:
        out['AgeAtSale'] = out['YrSold'] - out['YearBuilt']
        out['RemodAgeAtSale'] = out['YrSold'] - out['YearRemodAdd']
    out['TotalBath'] = out.get('FullBath',0) + 0.5*out.get('HalfBath',0) + out.get('BsmtFullBath',0) + 0.5*out.get('BsmtHalfBath',0)
    out['HasGarage'] = (out.get('GarageArea',0) > 0).astype(int)
    out['HasFireplace'] = (out.get('Fireplaces',0) > 0).astype(int)
    # Extra safe features
    out['RoomsPer100m2'] = np.where(out.get('GrLivArea',0)>0, out.get('TotRmsAbvGrd',0)/out.get('GrLivArea',1)*100, 0)
    out['IsRemodeled'] = (out.get('YearRemodAdd',0) > out.get('YearBuilt',0)).astype(int)
    out['LotAreaLog'] = np.log1p(out.get('LotArea',0))
    return out

FEATURES_ALL = SAFE_NUM_FEATURES + SAFE_CAT_FEATURES + [
    'TotalSF','AgeAtSale','RemodAgeAtSale','TotalBath','HasGarage','HasFireplace',
    'RoomsPer100m2','IsRemodeled','LotAreaLog'
]
