import numpy as np
import pandas as pd
from src.config import DATA_PATH
from src.preprocessing import HousingPreprocessor, remove_outliers


def test_no_missing_after_preprocessing():
    df = remove_outliers(pd.read_csv(DATA_PATH))
    out = HousingPreprocessor().fit_transform(df.drop(columns="SalePrice"))
    assert out.isna().sum().sum() == 0
    assert {"TotalSF", "TotalBath", "HouseAge"} <= set(out.columns)


def test_outliers_removed():
    df = pd.read_csv(DATA_PATH)
    assert len(remove_outliers(df)) < len(df)


def test_unseen_row_transforms():
    df = pd.read_csv(DATA_PATH)
    prep = HousingPreprocessor().fit(df.drop(columns="SalePrice"))
    row = df.drop(columns="SalePrice").iloc[[0]].copy()
    row["LotFrontage"] = np.nan
    assert prep.transform(row).shape[0] == 1
