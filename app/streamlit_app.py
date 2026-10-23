"""Streamlit app: real-time house price estimates + feature importance."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # so the pickled pipeline can import src.*
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "model.joblib")
    cfg = json.load(open(MODEL_DIR / "app_config.json"))
    metrics = json.load(open(MODEL_DIR / "metrics.json"))
    fi = pd.read_csv(MODEL_DIR / "feature_importance.csv", index_col=0)["importance"]
    return model, cfg, metrics, fi


model, cfg, metrics, fi = load_artifacts()
defaults, rng = cfg["defaults"], cfg["ranges"]

st.title("🏠 House Price Estimator")
st.caption(f"Model: **{metrics['best_model']}** · test R² = {metrics['test_r2']:.3f} · "
           f"typical error ≈ ${metrics['test_mae_usd']:,.0f} · trained on Ames, Iowa sales (USD)")

# ---------------- sidebar inputs ----------------
with st.sidebar:
    st.header("Property details")
    neighborhood = st.selectbox("Neighborhood", cfg["neighborhoods"],
                                index=cfg["neighborhoods"].index(defaults["Neighborhood"]))
    overall_qual = st.slider("Overall quality (1–10)", 1, 10, int(defaults["OverallQual"]))
    kitchen = st.select_slider("Kitchen quality", cfg["kitchen_quality"],
                               value=defaults["KitchenQual"])
    year_built = st.slider("Year built", int(rng["YearBuilt"][0]), 2010, int(defaults["YearBuilt"]))
    lot_area = st.number_input("Lot area (sq ft)", 1000, 100000, int(defaults["LotArea"]), 500)
    first = st.number_input("1st floor (sq ft)", 300, 4000, int(defaults["1stFlrSF"]), 50)
    second = st.number_input("2nd floor (sq ft)", 0, 2500, int(defaults["2ndFlrSF"]), 50)
    bsmt = st.number_input("Basement (sq ft)", 0, 4000, int(defaults["TotalBsmtSF"]), 50)
    beds = st.slider("Bedrooms above ground", 0, 8, int(defaults["BedroomAbvGr"]))
    baths = st.slider("Full bathrooms", 0, 4, int(defaults["FullBath"]))
    garage = st.slider("Garage capacity (cars)", 0, 4, int(defaults["GarageCars"]))
    fireplaces = st.slider("Fireplaces", 0, 3, int(defaults["Fireplaces"]))

# ---------------- build one full raw row ----------------
row = {k: (np.nan if v is None else v) for k, v in defaults.items()}
row.update({
    "Neighborhood": neighborhood, "OverallQual": overall_qual, "KitchenQual": kitchen,
    "YearBuilt": year_built, "LotArea": lot_area, "1stFlrSF": first, "2ndFlrSF": second,
    "GrLivArea": first + second, "TotalBsmtSF": bsmt, "BedroomAbvGr": beds,
    "FullBath": baths, "GarageCars": garage, "GarageArea": garage * 250,
    "Fireplaces": fireplaces, "YearRemodAdd": max(year_built, int(defaults["YearRemodAdd"])),
    "MSSubClass": str(int(float(defaults["MSSubClass"]))),
})
X = pd.DataFrame([row])

pred_log = float(model.predict(X)[0])
price = float(np.expm1(pred_log))
band = 1.2816 * metrics["test_rmse_log"]            # ~80% interval on the log scale
low, high = float(np.expm1(pred_log - band)), float(np.expm1(pred_log + band))

c1, c2, c3 = st.columns(3)
c1.metric("Estimated price", f"${price:,.0f}")
c2.metric("Likely range (80%)", f"${low:,.0f} – ${high:,.0f}")
c3.metric("Price per sq ft", f"${price / (first + second):,.0f}")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("What drives prices overall")
    st.bar_chart(fi.head(15).iloc[::-1], horizontal=True)
    st.caption("Share of the model's predictive power attributed to each feature.")
with right:
    st.subheader("Sensitivity of this estimate")
    st.caption("How the estimate changes if one thing improves.")
    rows = []
    for label, change in [
        ("Quality +1", {"OverallQual": min(overall_qual + 1, 10)}),
        ("Quality −1", {"OverallQual": max(overall_qual - 1, 1)}),
        ("+1 garage space", {"GarageCars": min(garage + 1, 4), "GarageArea": min(garage + 1, 4) * 250}),
        ("+1 full bathroom", {"FullBath": baths + 1}),
        ("+200 sq ft 2nd floor", {"2ndFlrSF": second + 200, "GrLivArea": first + second + 200}),
    ]:
        p = float(np.expm1(model.predict(pd.DataFrame([{**row, **change}]))[0]))
        rows.append({"Scenario": label, "Price": f"${p:,.0f}", "Δ vs current": f"{p - price:+,.0f}"})
    st.table(pd.DataFrame(rows))

with st.expander("Model benchmark"):
    comp = ROOT / "reports" / "model_comparison.csv"
    if comp.exists():
        st.dataframe(pd.read_csv(comp).drop(columns="best_params"), hide_index=True)
