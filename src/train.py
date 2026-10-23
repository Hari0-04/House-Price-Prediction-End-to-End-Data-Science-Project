"""Benchmark Linear / Ridge / Lasso / XGBoost with CV + GridSearchCV, save best."""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.config import (CV_FOLDS, DATA_PATH, FIG_DIR, MODEL_DIR, REPORT_DIR, SEED,
                        TARGET, TEST_SIZE)
from src.preprocessing import (HousingPreprocessor, is_categorical, is_numeric,
                               remove_outliers)


def make_pipeline(model):
    encode = ColumnTransformer([
        ("num", StandardScaler(), is_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), is_categorical),
    ])
    return Pipeline([("prep", HousingPreprocessor()), ("encode", encode), ("model", model)])


def candidates():
    return {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": [1, 3, 10, 30, 100]}),
        "Lasso": (Lasso(max_iter=20000),
                  {"model__alpha": [0.0003, 0.001, 0.003, 0.01]}),
        "XGBoost": (
            XGBRegressor(objective="reg:squarederror", random_state=SEED,
                         tree_method="hist", n_jobs=1),
            {"model__n_estimators": [300, 600],
             "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [3, 4],
             "model__subsample": [0.8],
             "model__colsample_bytree": [0.5]},
        ),
    }


def feature_importance(pipe, top=25):
    names = pipe.named_steps["encode"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    model = pipe.named_steps["model"]
    vals = (model.feature_importances_ if hasattr(model, "feature_importances_")
            else np.abs(model.coef_))
    s = pd.Series(vals, index=names).sort_values(ascending=False)
    return (s / s.sum()).head(top)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    print(f"Loaded {raw.shape}")
    df = remove_outliers(raw)
    print(f"After outlier removal: {df.shape}")

    X, y = df.drop(columns=TARGET), np.log1p(df[TARGET])   # model log-price
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    cv = KFold(CV_FOLDS, shuffle=True, random_state=SEED)

    results, fitted = [], {}
    for name, (model, grid) in candidates().items():
        t0 = time.time()
        pipe = make_pipeline(model)
        if grid:
            search = GridSearchCV(pipe, grid, cv=cv, scoring="r2", n_jobs=-1, refit=True)
            search.fit(X_tr, y_tr)
            best, cv_r2, params = search.best_estimator_, search.best_score_, search.best_params_
        else:
            cv_r2 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="r2").mean()
            best, params = pipe.fit(X_tr, y_tr), {}
        pred = best.predict(X_te)
        rmse_log = float(np.sqrt(mean_squared_error(y_te, pred)))
        results.append({
            "model": name,
            "cv_r2": round(float(cv_r2), 4),
            "test_r2": round(float(r2_score(y_te, pred)), 4),
            "test_rmse_log": round(rmse_log, 4),
            "test_rmse_usd": round(float(np.sqrt(mean_squared_error(np.expm1(y_te), np.expm1(pred)))), 0),
            "test_mae_usd": round(float(mean_absolute_error(np.expm1(y_te), np.expm1(pred))), 0),
            "best_params": {k.replace("model__", ""): v for k, v in params.items()},
            "seconds": round(time.time() - t0, 1),
        })
        fitted[name] = best
        print(f"{name:18s} CV R2={cv_r2:.4f} | test R2={results[-1]['test_r2']:.4f} "
              f"| {results[-1]['seconds']}s")

    res = pd.DataFrame(results)
    res.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = res.sort_values("cv_r2", ascending=False).iloc[0]["model"]
    best = fitted[best_name]
    lin = res.set_index("model").loc["Linear Regression"]
    top = res.set_index("model").loc[best_name]

    # Refit the winner on 100% of the data for deployment
    final = make_pipeline(candidates()[best_name][0].set_params(
        **{k: v for k, v in top["best_params"].items()}))
    final.fit(X, y)

    metrics = {
        "best_model": best_name,
        "test_r2": float(top["test_r2"]),
        "test_rmse_log": float(top["test_rmse_log"]),
        "test_rmse_usd": float(top["test_rmse_usd"]),
        "test_mae_usd": float(top["test_mae_usd"]),
        "linear_baseline_r2": float(lin["test_r2"]),
        "r2_improvement_vs_linear_pct": round(
            100 * (top["test_r2"] - lin["test_r2"]) / abs(lin["test_r2"]), 1),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    }
    json.dump(metrics, open(MODEL_DIR / "metrics.json", "w"), indent=2)

    # Artifacts used by the Streamlit app
    joblib.dump(final, MODEL_DIR / "model.joblib")
    fi = feature_importance(final)
    fi.to_csv(MODEL_DIR / "feature_importance.csv", header=["importance"])

    defaults = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            defaults[c] = None if X[c].isna().all() else float(X[c].median())
        else:
            v = X[c].value_counts(dropna=False).index[0]
            defaults[c] = None if pd.isna(v) else str(v)
    json.dump({
        "defaults": defaults,
        "neighborhoods": sorted(X["Neighborhood"].unique().tolist()),
        "kitchen_quality": ["Po", "Fa", "TA", "Gd", "Ex"],
        "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in
                   ["LotArea", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "YearBuilt"]},
    }, open(MODEL_DIR / "app_config.json", "w"), indent=2)

    # Figures
    plt.figure(figsize=(7, 4))
    plt.bar(res["model"], res["test_r2"], color=["#999", "#4C72B0", "#55A868", "#C44E52"])
    plt.ylim(max(0, res["test_r2"].min() - 0.05), 1)
    for i, v in enumerate(res["test_r2"]):
        plt.text(i, v + 0.003, f"{v:.3f}", ha="center")
    plt.ylabel("Test R²"); plt.title("Model comparison")
    plt.tight_layout(); plt.savefig(FIG_DIR / "08_model_comparison.png", dpi=130); plt.close()

    pred = best.predict(X_te)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(np.expm1(y_te), np.expm1(pred), alpha=0.5, s=14)
    lim = [np.expm1(y_te).min(), np.expm1(y_te).max()]
    ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted")
    ax[0].set_title(f"{best_name}: actual vs predicted")
    ax[1].scatter(pred, y_te - pred, alpha=0.5, s=14); ax[1].axhline(0, color="r", ls="--")
    ax[1].set_xlabel("Predicted log price"); ax[1].set_ylabel("Residual"); ax[1].set_title("Residuals")
    plt.tight_layout(); plt.savefig(FIG_DIR / "09_predictions_residuals.png", dpi=130); plt.close()

    plt.figure(figsize=(8, 6))
    fi.head(15).iloc[::-1].plot.barh(color="#4C72B0")
    plt.title("Top 15 drivers of price (feature importance)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "10_feature_importance.png", dpi=130); plt.close()

    print("\n", res.drop(columns="best_params").to_string(index=False))
    print("\n", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
