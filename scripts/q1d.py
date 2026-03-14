"""
research question 1 part d

UK/US resists AI because they are skeptical (or value humans more than AI)

tested with q6_q with kruskal wallis + mann whitney applied pairwise us vs. all and uk vs. all
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import build_parser, ensure_outdir, save_csv

COUNTRY_COL = "country"
SKEPTIC_COL = "Q6_grid_17" 
DK_CODES = {99}
VALID = {1, 2, 3, 4, 5}

US_CODE = "us"
UK_CODE = "uk"

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

def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("__NA__", np.nan), errors="coerce")

def run_country_greater_than_all(df, focal_country_code, label, country_col, skeptic_col):
    x = clean_numeric(df[skeptic_col])
    x = x.where(~x.isin(DK_CODES), np.nan)
    x = x.where(x.isin(list(VALID)), np.nan)

    df2 = df.loc[~x.isna(), [country_col]].copy()
    df2["skeptic"] = x.dropna().values

    valid_values = [1, 2, 3, 4, 5]

    print(f"\n===== {label}: {focal_country_code.upper()} more skeptical (higher Q6_Q) =====")
    print("N by country (non-missing skepticism):")
    print(df2[country_col].value_counts())

    groups = [sub["skeptic"].values for _, sub in df2.groupby(country_col)]
    kw_stat, kw_p = stats.kruskal(*groups)
    print(f"\nKruskal–Wallis H={kw_stat:.3f}, p={kw_p:.3e}")

    focal = df2[df2[country_col] == focal_country_code]["skeptic"].values
    if len(focal) == 0:
        raise ValueError(f"No rows found for focal country code '{focal_country_code}'. Check country codes.")

    results = []
    pvals = []

    for c in sorted(df2[country_col].unique()):
        if c == focal_country_code:
            continue
        other = df2[df2[country_col] == c]["skeptic"].values
        if len(other) == 0:
            continue

        U, p = stats.mannwhitneyu(focal, other, alternative="greater")
        pvals.append(p)

        n1, n2 = len(focal), len(other)

        ps_higher = U / (n1 * n2)

        delta = cliffs_delta_ordinal(focal, other, valid_values=valid_values)
        ps_from_delta = (delta + 1) / 2

        results.append({
            "compare": f"{focal_country_code} > {c}",
            "n_focal": n1,
            "n_other": n2,
            "U": float(U),
            "p_raw": float(p),
            "ps_higher": float(ps_higher),
            "cliffs_delta": float(delta), 
            "ps_from_delta": float(ps_from_delta)
        })

    rej, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
    for i, row in enumerate(results):
        row["p_fdr"] = float(p_adj[i])
        row["sig_fdr_0.05"] = bool(rej[i])

    out = pd.DataFrame(results).sort_values("p_fdr")
    print("\nPairwise MWU (one-sided, focal HIGHER), FDR-corrected:")
    print(out[[
        "compare",
        "cliffs_delta", "ps_from_delta",
        "p_raw", "p_fdr", "sig_fdr_0.05"
    ]].to_string(index=False))
    return df2, kw_stat, kw_p, out


def main():
    parser = build_parser(__doc__.strip().splitlines()[0], __file__)
    parser.add_argument("--country_col", default=COUNTRY_COL)
    parser.add_argument("--skeptic_col", default=SKEPTIC_COL)
    parser.add_argument("--us_code", default=US_CODE)
    parser.add_argument("--uk_code", default=UK_CODE)
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)

    for label, focal in [("H1d-US", args.us_code), ("H1d-UK", args.uk_code)]:
        df2, kw_stat, kw_p, out = run_country_greater_than_all(
            df, focal, label=label, country_col=args.country_col, skeptic_col=args.skeptic_col
        )
        prefix = label.lower().replace("-", "_")
        n_by_country = (
            df2[args.country_col]
            .value_counts()
            .rename_axis("country")
            .reset_index(name="n_non_missing_skepticism")
        )
        save_csv(n_by_country, outdir / f"{prefix}_n_by_country.csv")
        save_csv(pd.DataFrame([{"H": float(kw_stat), "p": float(kw_p)}]), outdir / f"{prefix}_kruskal_wallis.csv")
        save_csv(out, outdir / f"{prefix}_pairwise.csv")

    print(f"\nSaved outputs to: {Path(outdir)}")


if __name__ == "__main__":
    main()
