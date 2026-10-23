# House Price Prediction: End-to-End Data Science Project

Predict residential sale prices (Ames, Iowa) from 80 raw features, benchmark four
regression models, and serve estimates through a Streamlit app.

**Dataset:** Kaggle *House Prices: Advanced Regression Techniques* `train.csv`
(1,460 rows × 81 columns incl. target) → `data/raw/train.csv`.

## Project structure
```
data/raw/train.csv          raw data
src/config.py               paths, seed, CV settings
src/preprocessing.py        cleaning, imputation, feature engineering, outlier removal
src/eda.py                  EDA → reports/figures/01..07
src/train.py                CV + GridSearchCV benchmark, saves model + app artifacts
app/streamlit_app.py        real-time predictions + feature importance
tests/                      preprocessing tests
models/                     model.joblib, metrics.json, feature_importance.csv, app_config.json
reports/                    model_comparison.csv, figures
```

## Run it
```bash
pip install -r requirements.txt
python -m src.eda                 # figures
python -m src.train               # ~40 s on 1 CPU, faster with more cores
streamlit run app/streamlit_app.py
pytest tests
```

## Pipeline
**Cleaning & imputation**
- `NA` in Alley, Bsmt*, Garage*, Fireplace, Pool, Fence, MiscFeature means *feature absent* → `"None"` (not true missingness).
- Absent-feature numerics (garage/basement areas, baths, MasVnrArea) → 0.
- `LotFrontage` → median of the same neighbourhood (learned on the training fold only).
- Remaining gaps → median / mode learned in `fit()`; no train/test leakage.

**Outliers**: documented Ames outliers (GrLivArea > 4000 with price < $300k) removed from training; IQR report for the rest.

**Encoding**: quality scales (Po…Ex) → ordinal 0–5; `MSSubClass`, `MoSold`, `YrSold` treated as categories; other categoricals one-hot encoded.

**Transformation**: target modelled as `log1p(SalePrice)`; skewed continuous features (|skew| > 0.75) get `log1p`; numerics standardised.

**Engineered features**: `TotalSF`, `TotalBath`, `TotalPorchSF`, `HouseAge`, `RemodAge`, `IsRemodeled`, `HasPool/Garage/Bsmt/Fireplace/2ndFlr`, `QualxSF`.

**Modelling**: Linear Regression, Ridge, Lasso, XGBoost; 5-fold CV + `GridSearchCV` (R²); 80/20 hold-out; winner refit on all data for the app.

## Results (hold-out test set, 292 houses)
| Model | CV R² | Test R² | RMSE ($) | MAE ($) |
|---|---|---|---|---|
| Linear Regression | 0.838 | 0.899 | 21,484 | 15,098 |
| Ridge (α=30) | 0.916 | 0.919 | 19,568 | 13,959 |
| Lasso (α=0.001) | 0.916 | 0.924 | 18,967 | 13,702 |
| **XGBoost** | **0.920** | 0.923 | **18,269** | **13,516** |

Top drivers (XGBoost importance): `QualxSF`, `ExterQual`, `OverallQual`, `TotalSF`, `GarageCars`, basement/kitchen quality, neighbourhood.

## Notes
- Prices are in 2006–2010 Ames, Iowa USD; the model should not be used for other markets without retraining.
- The app shows an ~80% range derived from hold-out error, and a what-if table (e.g. +1 quality, +1 garage space).
- Dataset © Kaggle / Dean De Cock; follow the competition's terms of use.
# House-Price-Prediction-End-to-End-Data-Science-Project
