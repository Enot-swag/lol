import pytest
import pandas as pd
import numpy as np
from src.data_preparation import handle_missing, remove_outliers_iqr

def test_handle_missing():
    df = pd.DataFrame({'a': [1, np.nan, 3], 'b': ['x', np.nan, 'z']})
    df_clean = handle_missing(df)
    assert df_clean.isnull().sum().sum() == 0

def test_remove_outliers_iqr():
    df = pd.DataFrame({'value': [1,2,3,100,200]})
    df_clipped = remove_outliers_iqr(df, ['value'])
    assert df_clipped['value'].max() < 100  # должно быть обрезано