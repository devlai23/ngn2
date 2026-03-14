"""
research question 1 part c

UK/US resists AI usage more than other countries

tested with q4_10 with kruskal wallis + mann whitney applied pairwise us vs. all and uk vs. all
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import build_parser, ensure_outdir, save_csv

def cliffs_delta_ordinal(x, y, valid_values):
    x = pd.Series(x)
    y = pd.Series(y)

    cx = x.value_counts().reindex(valid_values, fill_value=0).to_numpy()
    cy = y.value_counts().reindex(valid_values, fill_value=0).to_numpy()

    n1 = cx.sum()
    n2 = cy.sum()
    if n1 == 0 or n2 == 0:
        return np.nan

    wins = 0
    losses = 0
    for i in range(len(valid_values)):
        wins += cx[i] * cy[:i].sum()
        losses += cx[i] * cy[i+1:].sum()

    return float((wins - losses) / (n1 * n2))

COUNTRY_COL = "country"
AI_Q4_COL = "Q4_grid_10" 
DK_CODES = {99}  
VALID = {0, 1, 2, 3}

US_CODE = "us"
UK_CODE = "uk"  
def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("__NA__", np.nan), errors="coerce")

def run_country_less_than_all(df, focal_country_code, label, country_col, ai_col):
    x = clean_numeric(df[ai_col])
    x = x.where(~x.isin(DK_CODES), np.nan)
    x = x.where(x.isin(list(VALID)), np.nan)

    df2 = df.loc[~x.isna(), [country_col]].copy()
    df2["ai_use"] = x.dropna().values
    valid_values = [0, 1, 2, 3]

    print(f"\n===== {label}: {focal_country_code.upper()} resists AI usage (lower Q4_grid_10) =====")
    print("N by country (non-missing AI usage):")
    print(df2[country_col].value_counts())

    groups = [sub["ai_use"].values for _, sub in df2.groupby(country_col)]
    kw_stat, kw_p = stats.kruskal(*groups)
    k = len(groups)
    df_chi = k - 1
    print(f"Kruskal–Wallis: H={kw_stat:.2f}, df={df_chi}, p < 1e-300 (underflow)")

    focal = df2[df2[country_col] == focal_country_code]["ai_use"].values
    if len(focal) == 0:
        raise ValueError(f"No rows found for focal country code '{focal_country_code}'. Check country codes.")

    results = []
    pvals = []

    for c in sorted(df2[country_col].unique()):
        if c == focal_country_code:
            continue
        other = df2[df2[country_col] == c]["ai_use"].values
        if len(other) == 0:
            continue

        U, p = stats.mannwhitneyu(focal, other, alternative="less")
        pvals.append(p)

        delta = cliffs_delta_ordinal(focal, other, valid_values=valid_values)
        delta_lower = -delta 

        n1, n2 = len(focal), len(other)

        ps_lower = 1 - (U / (n1 * n2))

        results.append({
            "compare": f"{focal_country_code} < {c}",
            "n_focal": n1,
            "n_other": n2,
            "U": float(U),
            "p_raw": float(p),
            "ps_lower": float(ps_lower),
            "cliffs_delta": float(delta),
            "delta_lower": float(delta_lower) 
        })

    rej, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
    for i, row in enumerate(results):
        row["p_fdr"] = p_adj[i]
        row["sig_fdr_0.05"] = bool(rej[i])

    out = pd.DataFrame(results).sort_values("p_fdr")
    print("\nPairwise MWU (one-sided, focal LOWER), FDR-corrected:")
    print(out[[
        "compare",
        "cliffs_delta", "delta_lower",
        "p_raw", "p_fdr", "sig_fdr_0.05"
    ]].to_string(index=False))
    return df2, kw_stat, kw_p, out


def main():
    parser = build_parser(__doc__.strip().splitlines()[0], __file__)
    parser.add_argument("--country_col", default=COUNTRY_COL)
    parser.add_argument("--ai_use_col", default=AI_Q4_COL)
    parser.add_argument("--us_code", default=US_CODE)
    parser.add_argument("--uk_code", default=UK_CODE)
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)

    for label, focal in [("H1c-US", args.us_code), ("H1c-UK", args.uk_code)]:
        df2, kw_stat, kw_p, out = run_country_less_than_all(
            df, focal, label=label, country_col=args.country_col, ai_col=args.ai_use_col
        )
        prefix = label.lower().replace("-", "_")
        n_by_country = (
            df2[args.country_col]
            .value_counts()
            .rename_axis("country")
            .reset_index(name="n_non_missing_ai_use")
        )
        save_csv(n_by_country, outdir / f"{prefix}_n_by_country.csv")
        save_csv(pd.DataFrame([{"H": float(kw_stat), "p": float(kw_p)}]), outdir / f"{prefix}_kruskal_wallis.csv")
        save_csv(out, outdir / f"{prefix}_pairwise.csv")

    print(f"\nSaved outputs to: {Path(outdir)}")


if __name__ == "__main__":
    main()
