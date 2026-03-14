"""
research question 3 part a

I find social media creators/personalities are helpful for....

pairwise comparisons between nigeria to all others, for all responses q20a-q20f, q21a, q21f

using kruskal wallis and cliffs delta
"""

import argparse
import os
from dataclasses import dataclass
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


def cliffs_delta_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return np.nan
    ps = u_stat / (n1 * n2)
    return float(2 * ps - 1)


@dataclass
class ItemSpec:
    name: str
    col: str
    valid_values: List[int]
    pretty: str


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


def stacked_ordinal_barplot(df: pd.DataFrame, country_col: str, value_col: str, valid_values: List[int], title: str, outpath: str) -> None:
    countries = sorted(df[country_col].dropna().unique().tolist())
    cats = valid_values

    counts = (
        df.dropna(subset=[value_col])
          .groupby([country_col, value_col])
          .size()
          .unstack(fill_value=0)
          .reindex(index=countries, columns=cats, fill_value=0)
    )

    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(countries))
    for v in cats:
        ax.bar(countries, pct[v].values, bottom=bottom, label=str(v))
        bottom += pct[v].values

    ax.set_title(title)
    ax.set_ylabel("Percent")
    ax.set_xlabel("Country")
    ax.legend(title="Response", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


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
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = read_csv_robust(args.data)
    df[args.country_col] = df[args.country_col].astype(str).str.lower().str.strip()

    q20_items = [
        ItemSpec("Q20_A", "Q20_grid_1", list(range(0, 5)), "Creators helpful: most important news"),
        ItemSpec("Q20_B", "Q20_grid_2", list(range(0, 5)), "Creators helpful: truthful/verifiable"),
        ItemSpec("Q20_C", "Q20_grid_3", list(range(0, 5)), "Creators helpful: different points of view"),
        ItemSpec("Q20_D", "Q20_grid_4", list(range(0, 5)), "Creators helpful: well-informed"),
        ItemSpec("Q20_E", "Q20_grid_5", list(range(0, 5)), "Creators helpful: interesting"),
        ItemSpec("Q20_F", "Q20_grid_6", list(range(0, 5)), "Creators helpful: relevant"),
    ]

    q21_items = [
        ItemSpec("Q21_A", "Q21_grid_1", list(range(0, 4)), "Frequency: get news from influencers/personalities"),
        ItemSpec("Q21_F", "Q21_grid_6", list(range(0, 4)), "Frequency: get news from journalists"),
    ]

    items = q20_items + q21_items
    countries = sorted(df[args.country_col].dropna().unique().tolist())

    all_kw_rows = []
    for item in items:
        if item.col not in df.columns:
            print(f"WARNING: missing column {item.col}; skipping {item.name}")
            continue

        tmp = df[[args.country_col, item.col]].copy()
        tmp[item.col] = clean_ordinal(tmp[item.col], valid_values=item.valid_values, dk_code=99)

        summary = summarize_by_country(tmp, args.country_col, item.col)
        summary.to_csv(os.path.join(args.outdir, f"{item.name}_country_summary.csv"), index=False)

        h, p = run_kw(tmp, args.country_col, item.col, countries)
        all_kw_rows.append({"item": item.name, "column": item.col, "H": h, "p": p})

        pairwise = run_pairwise_ng(tmp, args.country_col, item.col, args.nigeria_code, countries)
        pairwise.insert(0, "item", item.name)
        pairwise.insert(1, "column", item.col)
        pairwise.to_csv(os.path.join(args.outdir, f"{item.name}_pairwise_ng.csv"), index=False)

        stacked_ordinal_barplot(
            tmp, args.country_col, item.col, item.valid_values,
            title=f"{item.name}: {item.pretty}",
            outpath=os.path.join(args.outdir, f"{item.name}_stacked.png"),
        )

    kw_df = pd.DataFrame(all_kw_rows)
    if kw_df["p"].notna().any():
        kw_df["p_fdr_bh"] = bh_fdr(kw_df["p"].fillna(1.0).tolist())
    else:
        kw_df["p_fdr_bh"] = np.nan
    kw_df.to_csv(os.path.join(args.outdir, "all_items_kw.csv"), index=False)

    print(f"Done. Outputs in: {args.outdir}")


if __name__ == "__main__":
    main()
