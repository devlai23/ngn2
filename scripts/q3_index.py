"""
research question 3 part a

refer to q3_pairwise

this is a followup, using a “Creator Helpfulness Index” = mean(Q20 A-F) and did pairwise comparisons
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from common import default_output_dir


def bh_fdr(pvals: List[float]) -> np.ndarray:
    p = np.array(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def clean_ordinal(series: pd.Series, valid_values: List[int], dk_code: int = 99) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace(dk_code, np.nan)
    s = s.where(s.isin(valid_values), np.nan)
    return s


def cliffs_delta_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return np.nan
    ps = u_stat / (n1 * n2)
    return float(2 * ps - 1)


def summarize_by_country(df: pd.DataFrame, country_col: str, value_col: str) -> pd.DataFrame:
    def iqr(x: pd.Series) -> float:
        return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

    g = df.groupby(country_col)[value_col]
    out = pd.DataFrame({
        "n": g.count(),
        "median": g.median(),
        "q25": g.quantile(0.25),
        "q75": g.quantile(0.75),
        "iqr": g.apply(iqr),
        "mean": g.mean(),
        "std": g.std(ddof=1),
    }).reset_index()
    return out


def run_kw(df: pd.DataFrame, country_col: str, value_col: str, countries: List[str]) -> Tuple[float, float]:
    groups = []
    for c in countries:
        vals = df.loc[df[country_col] == c, value_col].dropna().to_numpy()
        groups.append(vals)
    if any(len(g) == 0 for g in groups):
        return np.nan, np.nan
    h, p = stats.kruskal(*groups, nan_policy="omit")
    return float(h), float(p)


def run_pairwise_ng(df: pd.DataFrame, country_col: str, value_col: str, nigeria_code: str, countries: List[str]) -> pd.DataFrame:
    ng = df.loc[df[country_col] == nigeria_code, value_col].dropna().to_numpy()
    rows = []
    for other in countries:
        if other == nigeria_code:
            continue
        ot = df.loc[df[country_col] == other, value_col].dropna().to_numpy()
        if len(ng) == 0 or len(ot) == 0:
            rows.append({"compare_to": other, "n_ng": len(ng), "n_other": len(ot), "u": np.nan, "p": np.nan, "cliffs_delta": np.nan})
            continue

        u, p = stats.mannwhitneyu(ng, ot, alternative="two-sided", method="auto")
        rows.append({
            "compare_to": other,
            "n_ng": len(ng),
            "n_other": len(ot),
            "u": float(u),
            "p": float(p),
            "cliffs_delta": cliffs_delta_from_u(u, len(ng), len(ot)),
        })

    out = pd.DataFrame(rows)
    if out["p"].notna().any():
        out["p_fdr_bh"] = bh_fdr(out["p"].fillna(1.0).tolist())
    else:
        out["p_fdr_bh"] = np.nan
    return out


def read_csv_robust(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        na_values=["__NA__", "NA", "N/A", ""],
        keep_default_na=True,
        low_memory=False,
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to combined_coded_responses.csv")
    parser.add_argument("--outdir", default=str(default_output_dir(__file__)), help="Output directory")
    parser.add_argument("--country_col", default="country")
    parser.add_argument("--nigeria_code", default="ng")
    parser.add_argument("--min_items", type=int, default=4, help="Minimum answered Q20 items to compute index")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = read_csv_robust(args.data)
    df[args.country_col] = df[args.country_col].astype(str).str.lower().str.strip()

    q20_cols = [f"Q20_grid_{i}" for i in range(1, 7)]
    for c in q20_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    q20_clean = df[q20_cols].apply(lambda s: clean_ordinal(s, valid_values=list(range(0, 5)), dk_code=99))

    answered = q20_clean.notna().sum(axis=1)
    index = q20_clean.mean(axis=1)
    index = index.where(answered >= args.min_items, np.nan)

    tmp = pd.DataFrame({
        args.country_col: df[args.country_col],
        "creator_helpfulness_index": index,
        "q20_items_answered": answered,
    })

    countries = sorted(tmp[args.country_col].dropna().unique().tolist())

    summary = summarize_by_country(tmp, args.country_col, "creator_helpfulness_index")
    summary.to_csv(os.path.join(args.outdir, "index_country_summary.csv"), index=False)

    h, p = run_kw(tmp, args.country_col, "creator_helpfulness_index", countries)
    pd.DataFrame([{"H": h, "p": p}]).to_csv(os.path.join(args.outdir, "index_kw.csv"), index=False)

    pairwise = run_pairwise_ng(tmp, args.country_col, "creator_helpfulness_index", args.nigeria_code, countries)
    pairwise.to_csv(os.path.join(args.outdir, "index_pairwise_ng.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [tmp.loc[tmp[args.country_col] == c, "creator_helpfulness_index"].dropna().to_numpy() for c in countries]
    ax.boxplot(data, labels=countries, showfliers=False)
    ax.set_title("Creator Helpfulness Index (mean of Q20 A–F)")
    ax.set_xlabel("Country")
    ax.set_ylabel("Index (0–4)")
    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, "index_boxplot.png"), dpi=200)
    plt.close(fig)

    print(f"Done. Outputs in: {args.outdir}")


if __name__ == "__main__":
    main()
