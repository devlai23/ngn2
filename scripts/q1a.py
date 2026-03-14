"""
research question 1 part a

In Nigeria, people are more comfortable using AI, while also doing more fact-checking. Whereas, in the UK and US, AI usage is 
resisted because they are more skeptical of its accuracy

tested with q6_r with kruskal wallis + mann whitney
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
COMFORT_COL = "Q6_grid_18" 
NIGERIA_CODE = "ng" 
DK_CODES = {99}  

def main():
    parser = build_parser(__doc__.strip().splitlines()[0], __file__)
    parser.add_argument("--country_col", default=COUNTRY_COL)
    parser.add_argument("--comfort_col", default=COMFORT_COL)
    parser.add_argument("--nigeria_code", default=NIGERIA_CODE)
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)

    x = pd.to_numeric(df[args.comfort_col].replace("__NA__", np.nan), errors="coerce")
    x = x.where(~x.isin(DK_CODES), np.nan)
    x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
    df2 = df.loc[~x.isna(), [args.country_col]].copy()
    df2["comfort"] = x.dropna().values

    n_by_country = (
        df2[args.country_col]
        .value_counts()
        .rename_axis("country")
        .reset_index(name="n_non_missing_comfort")
    )
    save_csv(n_by_country, outdir / "comfort_n_by_country.csv")
    print("N by country (non-missing comfort):")
    print(df2[args.country_col].value_counts())

    groups = [sub["comfort"].values for _, sub in df2.groupby(args.country_col)]
    kw_stat, kw_p = stats.kruskal(*groups)
    kw_df = pd.DataFrame([{"H": float(kw_stat), "p": float(kw_p)}])
    save_csv(kw_df, outdir / "comfort_kruskal_wallis.csv")
    print(f"\nKruskal–Wallis H={kw_stat:.3f}, p={kw_p:.4g}")

    ng = df2[df2[args.country_col] == args.nigeria_code]["comfort"].values
    if len(ng) == 0:
        raise ValueError(f"No rows found for Nigeria code '{args.nigeria_code}'. Check your country codes.")

    results = []
    pvals = []
    valid_values = [1, 2, 3, 4, 5]

    for c in sorted(df2[args.country_col].unique()):
        if c == args.nigeria_code:
            continue
        other = df2[df2[args.country_col] == c]["comfort"].values
        if len(other) == 0:
            continue

        u_stat, p_val = stats.mannwhitneyu(ng, other, alternative="greater")
        pvals.append(p_val)

        delta = cliffs_delta_ordinal(ng, other, valid_values=valid_values)
        ps_from_delta = (delta + 1) / 2

        results.append({
            "compare": f"{args.nigeria_code} > {c}",
            "n_ng": len(ng),
            "n_other": len(other),
            "U": float(u_stat),
            "p_raw": float(p_val),
            "cliffs_delta": float(delta),
            "ps_from_delta": float(ps_from_delta),
        })

    rej, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
    for i, row in enumerate(results):
        row["p_fdr"] = float(p_adj[i])
        row["sig_fdr_0.05"] = bool(rej[i])

    out = pd.DataFrame(results).sort_values("p_fdr")
    save_csv(out, outdir / "comfort_pairwise_ng_vs_others.csv")
    print("\nNigeria vs others (one-sided MWU), FDR-corrected:")
    print(out[[
        "compare",
        "cliffs_delta", "ps_from_delta",
        "p_raw", "p_fdr", "sig_fdr_0.05"
    ]].to_string(index=False))
    print(f"\nSaved outputs to: {Path(outdir)}")


if __name__ == "__main__":
    main()
