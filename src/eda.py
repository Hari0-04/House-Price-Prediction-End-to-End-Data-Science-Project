"""Exploratory analysis -> figures in reports/figures and a console summary."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DATA_PATH, FIG_DIR, TARGET
from src.preprocessing import iqr_outlier_report

sns.set_theme(style="whitegrid")


def save(name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=130)
    plt.close()


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")

    # 1. Missing values
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0] * 100
    plt.figure(figsize=(8, 6))
    sns.barplot(x=miss.values, y=miss.index, color="#4C72B0")
    plt.xlabel("% missing")
    plt.title("Missing values by feature")
    save("01_missing_values")

    # 2. Target distribution: raw vs log
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], kde=True, ax=ax[0]).set_title(
        f"SalePrice (skew={df[TARGET].skew():.2f})")
    sns.histplot(np.log1p(df[TARGET]), kde=True, ax=ax[1], color="green").set_title(
        f"log1p(SalePrice) (skew={np.log1p(df[TARGET]).skew():.2f})")
    save("02_target_distribution")

    # 3. Correlation with target
    num = df.select_dtypes(include=np.number).drop(columns="Id")
    corr = num.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    top = corr.head(15)
    plt.figure(figsize=(7, 6))
    sns.barplot(x=top.values, y=top.index, color="#55A868")
    plt.title("Top 15 numeric correlations with SalePrice")
    save("03_top_correlations")
    print("\nTop correlations:\n", top.round(3).to_string())

    # 4. Heatmap of the top features
    cols = corr.head(10).index.tolist() + [TARGET]
    plt.figure(figsize=(9, 7))
    sns.heatmap(num[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (top 10 features)")
    save("04_correlation_heatmap")

    # 5. Outliers: GrLivArea vs price
    out = (df["GrLivArea"] > 4000) & (df[TARGET] < 300_000)
    plt.figure(figsize=(7, 5))
    plt.scatter(df.loc[~out, "GrLivArea"], df.loc[~out, TARGET], alpha=0.5, s=14)
    plt.scatter(df.loc[out, "GrLivArea"], df.loc[out, TARGET], color="red", s=40,
                label=f"outliers removed ({out.sum()})")
    plt.xlabel("GrLivArea"); plt.ylabel("SalePrice"); plt.legend()
    plt.title("Above-ground living area vs price")
    save("05_outliers_grlivarea")

    # 6. Quality & neighbourhood
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="OverallQual", y=TARGET, color="#8172B2")
    plt.title("Price by overall quality")
    save("06_price_by_quality")

    order = df.groupby("Neighborhood")[TARGET].median().sort_values().index
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Neighborhood", y=TARGET, order=order, color="#C44E52")
    plt.xticks(rotation=70)
    plt.title("Price by neighbourhood")
    save("07_price_by_neighborhood")

    # 7. Statistical check: skewness + IQR outlier counts
    print("\nIQR outlier counts:\n",
          iqr_outlier_report(num, ["LotArea", "GrLivArea", "TotalBsmtSF", TARGET]).to_string(index=False))
    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
