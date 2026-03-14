"""
research question 1 part e

Nigeria often uses an AI tools to learn more about a topic

tested with q12_3 with chi squared
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

from common import build_parser, ensure_outdir, save_csv


def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("__NA__", np.nan), errors="coerce")


def make_ai_tool_binary(q12: pd.Series) -> pd.Series:
    x = clean_numeric(q12)
    x = x.where(x.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.nan)
    x = x.where(x != 9, np.nan) 
    return np.where(x.isna(), np.nan, (x == 3).astype(int))


def pct_yes(bin01: pd.Series) -> tuple[float, int, int]:
    valid = pd.Series(bin01).dropna()
    n = int(valid.shape[0])
    if n == 0:
        return np.nan, 0, 0
    yes = int((valid == 1).sum())
    return (yes / n) * 100.0, yes, n


def phi_from_2x2(ct_2x2: np.ndarray, chi2_stat: float) -> float:
    n = ct_2x2.sum()
    if n <= 0:
        return 0.0
    return float(np.sqrt(chi2_stat / n))


def main():
    ap = build_parser(__doc__.strip().splitlines()[0], __file__)
    ap.add_argument("--country_col", default="country")
    ap.add_argument("--q12_col", default="Q12", help="Column containing Q12 select-one code")
    ap.add_argument("--nigeria_code", default="ng")
    ap.add_argument("--min_expected_for_chi2", type=float, default=5.0,
                    help="If any expected cell in 2x2 < this, use Fisher's exact instead of chi-square")
    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)

    if args.country_col not in df.columns:
        raise ValueError(f"Missing country column: {args.country_col}")
    if args.q12_col not in df.columns:
        raise ValueError(f"Missing Q12 column: {args.q12_col}")

    df = df.copy()
    df[args.country_col] = df[args.country_col].astype(str).str.lower().str.strip()

    df["ai_tool"] = make_ai_tool_binary(df[args.q12_col])

    tmp = df.dropna(subset=[args.country_col, "ai_tool"]).copy()
    tmp["ai_tool"] = tmp["ai_tool"].astype(int)

    ct_all = pd.crosstab(tmp[args.country_col], tmp["ai_tool"]).reindex(columns=[0, 1], fill_value=0)

    print("Overall contingency table (counts): country x ai_tool(0/1)")
    print(ct_all)

    chi2_stat, p, dof, expected = chi2_contingency(ct_all)
    n_all = ct_all.to_numpy().sum()
    cramers_v = float(np.sqrt(chi2_stat / (n_all * 1))) if n_all > 0 else 0.0

    print("\nOverall chi-square test of independence")
    print(f"chi2 = {chi2_stat:.4f}")
    print(f"dof  = {dof}")
    print(f"p    = {p:.4e}")
    print(f"Cramér's V = {cramers_v:.4f}")

    expected_df = pd.DataFrame(expected, index=ct_all.index, columns=ct_all.columns)
    low_expected = int((expected < args.min_expected_for_chi2).sum())
    total_cells = expected.size
    print(f"Cells with expected count < {args.min_expected_for_chi2:g}: {low_expected}/{total_cells}")

    save_csv(ct_all.reset_index(), outdir / "q12_ai_tool_overall_counts.csv")
    overall_rowpct = (ct_all.div(ct_all.sum(axis=1), axis=0) * 100).round(2)
    save_csv(overall_rowpct.reset_index(), outdir / "q12_ai_tool_overall_rowpct.csv")
    save_csv(expected_df.reset_index(), outdir / "q12_ai_tool_expected_counts.csv")
    save_csv(pd.DataFrame([{
        "chi2": float(chi2_stat),
        "dof": int(dof),
        "p": float(p),
        "cramers_v": float(cramers_v),
        "low_expected_cells": low_expected,
        "total_cells": total_cells,
    }]), outdir / "q12_ai_tool_overall_test.csv")

    ng = tmp[tmp[args.country_col] == args.nigeria_code]
    if ng.empty:
        raise ValueError(f"No rows found for Nigeria code '{args.nigeria_code}'. Check country codes.")

    results = []
    pvals = []

    for other_code in sorted(tmp[args.country_col].unique()):
        if other_code == args.nigeria_code:
            continue
        other = tmp[tmp[args.country_col] == other_code]
        if other.empty:
            continue

        sub = pd.concat([ng, other], ignore_index=True)
        ct = pd.crosstab(sub[args.country_col], sub["ai_tool"]).reindex(
            index=[args.nigeria_code, other_code],
            columns=[0, 1],
            fill_value=0
        )

        ng_pct = (ct.loc[args.nigeria_code, 1] / ct.loc[args.nigeria_code].sum() * 100) if ct.loc[args.nigeria_code].sum() else np.nan
        other_pct = (ct.loc[other_code, 1] / ct.loc[other_code].sum() * 100) if ct.loc[other_code].sum() else np.nan

        chi2_s, p_s, dof_s, exp_s = chi2_contingency(ct)
        use_fisher = (exp_s < args.min_expected_for_chi2).any()

        if use_fisher:
            a = int(ct.loc[args.nigeria_code, 0])
            b = int(ct.loc[args.nigeria_code, 1])
            c = int(ct.loc[other_code, 0])
            d = int(ct.loc[other_code, 1])
            _, p_test = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            test_name = "Fisher"
            chi2_for_phi = chi2_s 
            phi = phi_from_2x2(ct.to_numpy(), chi2_for_phi)
        else:
            p_test = p_s
            test_name = "Chi-square"
            phi = phi_from_2x2(ct.to_numpy(), chi2_s)

        pvals.append(p_test)

        results.append({
            "compare": f"{args.nigeria_code} vs {other_code}",
            "test": test_name,
            "n_ng": int(ct.loc[args.nigeria_code].sum()),
            "n_other": int(ct.loc[other_code].sum()),
            "ng_pct_ai_tool": float(ng_pct),
            "other_pct_ai_tool": float(other_pct),
            "pct_point_diff_ng_minus_other": float(ng_pct - other_pct),
            "phi": float(phi),
            "p_raw": float(p_test),
        })

    rej, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
    for i, row in enumerate(results):
        row["p_fdr"] = float(p_adj[i])
        row["sig_fdr_0.05"] = bool(rej[i])

    out = pd.DataFrame(results).sort_values("p_fdr")

    print("\nNigeria vs others: selecting AI tool (Q12==3), FDR-corrected")
    print(out[[
        "compare", "n_ng", "n_other",
        "ng_pct_ai_tool", "other_pct_ai_tool", "pct_point_diff_ng_minus_other",
        "phi", "p_raw", "p_fdr", "sig_fdr_0.05"
    ]].to_string(index=False))

    save_csv(out, outdir / "q12_ai_tool_pairwise_ng_vs_others.csv")
    print(f"\nSaved outputs to: {Path(outdir)}")


if __name__ == "__main__":
    main()
