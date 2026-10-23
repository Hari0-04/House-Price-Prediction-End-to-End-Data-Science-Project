"""Leakage-safe cleaning + feature engineering as a scikit-learn transformer.

Everything that is *learned* from data (group medians, modes, skewed columns)
is learned in ``fit`` on the training fold only, so cross-validation and the
Streamlit app apply exactly the same logic.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# "NA" in these columns means "feature absent", not "missing data".
NONE_COLS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
# Numeric columns where missing means the feature does not exist -> 0
ZERO_COLS = [
    "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]
# Ordered quality scales -> integers (keeps the ordering, avoids dummy blow-up)
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
# Numeric codes that are really categories
AS_CATEGORY = ["MSSubClass", "MoSold", "YrSold"]
DROP_COLS = ["Id", "GarageYrBlt"]


class HousingPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=0.75):
        self.skew_threshold = skew_threshold

    # ---- stateless steps -------------------------------------------------
    def _clean(self, X):
        X = X.copy()
        X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])
        for c in NONE_COLS:
            if c in X.columns:
                X[c] = X[c].fillna("None")
        for c in ZERO_COLS:
            if c in X.columns:
                X[c] = X[c].fillna(0)
        # LotFrontage: median of the same neighbourhood (learned in fit)
        if "LotFrontage" in X.columns:
            fill = X["Neighborhood"].map(self.lot_median_).fillna(self.lot_global_)
            X["LotFrontage"] = X["LotFrontage"].fillna(fill)
        for c in QUALITY_COLS:
            if c in X.columns:
                X[c] = X[c].fillna("None").map(QUALITY_MAP).fillna(3).astype(int)
        for c in AS_CATEGORY:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c]).astype(int).astype(str)
        return X

    @staticmethod
    def _engineer(X):
        X = X.copy()
        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        X["TotalBath"] = (X["FullBath"] + 0.5 * X["HalfBath"]
                          + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"])
        X["TotalPorchSF"] = (X["OpenPorchSF"] + X["EnclosedPorch"]
                             + X["3SsnPorch"] + X["ScreenPorch"] + X["WoodDeckSF"])
        X["HouseAge"] = (X["YrSold"].astype(int) - X["YearBuilt"]).clip(lower=0)
        X["RemodAge"] = (X["YrSold"].astype(int) - X["YearRemodAdd"]).clip(lower=0)
        X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
        X["HasPool"] = (X["PoolArea"] > 0).astype(int)
        X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
        X["HasBsmt"] = (X["TotalBsmtSF"] > 0).astype(int)
        X["Has2ndFlr"] = (X["2ndFlrSF"] > 0).astype(int)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        X["QualxSF"] = X["OverallQual"] * X["GrLivArea"]
        return X

    # ---- sklearn API -----------------------------------------------------
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.lot_median_ = X.groupby("Neighborhood")["LotFrontage"].median()
        self.lot_global_ = X["LotFrontage"].median()

        Xc = self._engineer(self._clean(X))
        num = Xc.select_dtypes(include=np.number)
        cat = Xc.drop(columns=num.columns)
        self.medians_ = num.median()
        self.modes_ = cat.mode().iloc[0] if cat.shape[1] else pd.Series(dtype=object)

        # Skewed continuous columns -> log1p (skip binary / low-cardinality codes)
        cont = [c for c in num.columns if num[c].nunique() > 15]
        skew = num[cont].skew().abs()
        self.skewed_cols_ = skew[skew > self.skew_threshold].index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        Xc = self._engineer(self._clean(X))
        for c, v in self.medians_.items():
            if c in Xc.columns:
                Xc[c] = Xc[c].fillna(v)
        for c, v in self.modes_.items():
            if c in Xc.columns:
                Xc[c] = Xc[c].fillna(v)
        for c in self.skewed_cols_:
            Xc[c] = np.log1p(Xc[c].clip(lower=0))
        return Xc


def is_numeric(X: pd.DataFrame):
    """Column selector (importable, so the fitted pipeline can be unpickled anywhere)."""
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


def is_categorical(X: pd.DataFrame):
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the documented Ames outliers: huge living area but low price."""
    mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300_000)
    return df.loc[~mask].reset_index(drop=True)


def iqr_outlier_report(df: pd.DataFrame, cols, k=1.5) -> pd.DataFrame:
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({"feature": c, "n_outliers": n, "pct": round(100 * n / len(df), 2)})
    return pd.DataFrame(rows).sort_values("n_outliers", ascending=False)
