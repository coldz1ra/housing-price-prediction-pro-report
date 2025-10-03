import numpy as np, pandas as pd
from src.transformers import NeighborhoodStatsTransformer
def test_nbhd_transformer_shapes():
    X=pd.DataFrame({"Neighborhood":["A","A","B","C","C","C"]})
    y=np.log1p([100,120,130,80,90,110])
    t=NeighborhoodStatsTransformer(); t.fit(X,y); Z=t.transform(X)
    assert Z.shape==(6,3)
