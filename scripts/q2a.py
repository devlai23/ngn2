"""
research question 2 part a

YouTube is becoming a primary source across age groups

tested with q5_29 + data viz + cramers v for effect size
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from common import build_parser, ensure_outdir, save_csv

AGE_COL = "age"

PLATFORMS = {
    "YouTube": "Q5_m_29",
    "Instagram": "Q5_m_3",
    "X": "Q5_m_2",
    # "Snapchat": "Q5_m_7",
    "Facebook": "Q5_m_11",
    "TikTok": "Q5_m_14",
}

DK_CODES = {99}

def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("__NA__", np.nan), errors="coerce")


def selected_binary(s: pd.Series) -> pd.Series:
    x = clean_numeric(s)
    x = x.where(~x.isin(DK_CODES), np.nan)
    return x.map({1: 1.0, 2: 0.0})


def add_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[AGE_COL] = clean_numeric(out[AGE_COL])

    bins = [0, 25, 50, 120]
    labels = ["18-24", "25-49", "50+"]
    out["age_bin"] = pd.cut(out[AGE_COL], bins=bins, labels=labels, include_lowest=True)

    out["age_bin"] = out["age_bin"].astype(
        pd.CategoricalDtype(categories=labels, ordered=True)
    )
    return out


def platform_percent_by_age(df: pd.DataFrame, platforms: dict) -> pd.DataFrame:
    tmp = df.copy()

    valid_platforms = {}
    for name, col in platforms.items():
        if col not in tmp.columns:
            print(f"[warn] Missing column: {col} (skipping {name})")
            continue
        tmp[name] = selected_binary(tmp[col])
        valid_platforms[name] = col

    if "age_bin" not in tmp.columns:
        raise ValueError("age_bin column missing. Run add_age_bins() first.")

    mat = tmp.groupby("age_bin", observed=False)[list(valid_platforms.keys())].mean() * 100.0
    mat = mat.dropna(how="all") 
    return mat


def cramers_v(chi2_stat: float, n: int, r: int, k: int) -> float:
    denom = n * min(r - 1, k - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2_stat / denom))


def chi2_by_platform_age(df: pd.DataFrame, platforms: dict) -> pd.DataFrame:
    if "age_bin" not in df.columns:
        raise ValueError("age_bin column missing. Run add_age_bins() first.")

    rows = []
    pvals = []

    for name, col in platforms.items():
        if col not in df.columns:
            continue

        sel = selected_binary(df[col])
        tmp = df[["age_bin"]].copy()
        tmp["sel"] = sel
        tmp = tmp.dropna(subset=["age_bin", "sel"]).copy()
        tmp["sel"] = tmp["sel"].astype(int)

        ct = pd.crosstab(tmp["age_bin"], tmp["sel"]).reindex(columns=[0, 1], fill_value=0)

        if ct.shape[1] < 2 or (ct[1].sum() == 0) or (ct[0].sum() == 0):
            rows.append({
                "platform": name,
                "n_valid": int(ct.to_numpy().sum()),
                "chi2": np.nan,
                "dof": np.nan,
                "p_raw": np.nan,
                "cramers_v": np.nan,
                "pct_<25": np.nan,
                "pct_50+": np.nan,
                "delta_pp_<25_minus_50plus": np.nan,
                "min_expected": np.nan,
                "note": "No variance / insufficient variation for chi-square"
            })
            continue

        chi2_stat, p, dof, expected = chi2_contingency(ct)
        pvals.append(p)

        n = int(ct.to_numpy().sum())
        r, k = ct.shape
        v = cramers_v(chi2_stat, n, r, k)

        row_pct = ct.div(ct.sum(axis=1), axis=0) * 100.0
        pct_young = float(row_pct.loc["18-24", 1]) if "18-24" in row_pct.index else np.nan
        pct_old = float(row_pct.loc["50+", 1]) if "50+" in row_pct.index else np.nan

        delta_pp = pct_young - pct_old

        rows.append({
            "platform": name,
            "n_valid": n,
            "chi2": float(chi2_stat),
            "dof": int(dof),
            "p_raw": float(p),
            "cramers_v": float(v),
            "pct_<25": pct_young,
            "pct_50+": pct_old,
            "delta_pp_<25_minus_50plus": float(delta_pp),
            "min_expected": float(np.min(expected)),
            "note": ""
        })

    out = pd.DataFrame(rows)

    mask = out["p_raw"].notna()
    if mask.sum() > 0:
        rej, p_adj, _, _ = multipletests(out.loc[mask, "p_raw"].values, method="fdr_bh")
        out.loc[mask, "p_fdr"] = p_adj
        out.loc[mask, "sig_fdr_0.05"] = rej
    else:
        out["p_fdr"] = np.nan
        out["sig_fdr_0.05"] = False

    out = out.sort_values(["p_fdr", "p_raw"], na_position="last")
    return out

def plot_grouped_bars(mat: pd.DataFrame, outpath: str):
    plt.figure(figsize=(12, 6))

    x = np.arange(len(mat.index))
    n_platforms = len(mat.columns)
    width = 0.8 / n_platforms

    for i, col in enumerate(mat.columns):
        plt.bar(
            x + i * width - (0.8 - width) / 2, mat[col].values, width=width, label=col
        )

    plt.xticks(x, mat.index.astype(str))
    plt.ylabel("% selected (within age group)")
    plt.xlabel("Age group")
    plt.title("Q5 platform use for news by age group (video platforms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_platform_heatmap(mat: pd.DataFrame, outpath: str):
    data = mat.values.astype(float)

    plt.figure(figsize=(10, max(4, 0.7 * len(mat.index))))
    plt.imshow(data, aspect="auto")
    plt.colorbar(label="% selected (within age group)")

    plt.yticks(np.arange(mat.shape[0]), mat.index.astype(str))
    plt.xticks(np.arange(mat.shape[1]), mat.columns, rotation=30, ha="right")

    plt.title("Q5 platform use for news by age group (heatmap)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_small_multiples(mat: pd.DataFrame, outpath: str):
    n = len(mat.columns)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    plt.figure(figsize=(14, 3.8 * nrows))

    for i, col in enumerate(mat.columns, start=1):
        ax = plt.subplot(nrows, ncols, i)
        ax.bar(mat.index.astype(str), mat[col].values)
        ax.set_title(col)
        ax.set_ylim(0, max(100, np.nanmax(mat.values) * 1.1))
        ax.set_ylabel("% selected")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Q5 platform use for news by age group (small multiples)", y=0.995)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    parser = build_parser(__doc__.strip().splitlines()[0], __file__)
    args = parser.parse_args()
    outdir = ensure_outdir(args.outdir)

    df = pd.read_csv(args.csv)
    df = add_age_bins(df)

    mat = platform_percent_by_age(df, PLATFORMS)
    save_csv(mat.reset_index(), outdir / "q5_platform_percent_by_age.csv")

    print("\n% selected by age group (within age group denominator):")
    print(mat.round(1).to_string())

    plot_grouped_bars(mat, os.path.join(outdir, "Q5_platforms_by_age_grouped_bars.png"))
    plot_platform_heatmap(mat, os.path.join(outdir, "Q5_platforms_by_age_heatmap.png"))
    plot_small_multiples(
        mat, os.path.join(outdir, "Q5_platforms_by_age_small_multiples.png")
    )

    stats_df = chi2_by_platform_age(df, PLATFORMS)
    save_csv(stats_df, outdir / "Q5_platform_age_chi2_cramersv.csv")
    print("\nChi-square by platform (age_bin x selected), with effect size and direction:")
    print(stats_df[[
        "platform", "n_valid", "cramers_v",
        "pct_<25", "pct_50+", "delta_pp_<25_minus_50plus",
        "p_raw", "p_fdr", "sig_fdr_0.05", "min_expected", "note"
    ]].to_string(index=False))

    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
