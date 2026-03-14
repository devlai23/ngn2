"""
for testing the contradictions of selecting q16 option 14 and q16 option 15 simultaneously
with country to country analyses
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import build_parser, ensure_outdir, save_csv

Q16_N_COL = "Q16_grid_14"
Q16_O_COL = "Q16_grid_15"

def plot_country_age_grouped_bar(
    by_country_age: pd.DataFrame,
    value_col: str,
    out_png: str,
    title: str,
    ylabel: str,
):
    plot_df = (
        by_country_age.reset_index()
        .pivot(index="country", columns="age_bin", values=value_col)
        .sort_index()
    )

    desired_cols = ["18-29", "30–49", "50+"]
    plot_df = plot_df.reindex(columns=[c for c in desired_cols if c in plot_df.columns])

    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Country")
    ax.set_ylabel(ylabel)
    ax.legend(title="Age group", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved plot: {out_png}")

def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("__NA__", np.nan), errors="coerce")

def agree45(s: pd.Series) -> pd.Series:
    x = clean_numeric(s)
    x = x.where(~x.isin([99]), np.nan)  
    x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
    return np.where(x.isna(), np.nan, (x >= 4).astype(int))

def disagree12(s: pd.Series) -> pd.Series:
    x = clean_numeric(s)
    x = x.where(~x.isin([99]), np.nan)    
    x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
    return np.where(x.isna(), np.nan, (x <= 2).astype(int))

def add_age_bin(df: pd.DataFrame) -> pd.DataFrame:
    a = clean_numeric(df["age"])
    bins = [18, 30, 50, 120]
    labels = ["18-29", "30–49", "50+"]
    df = df.copy()
    df["age_bin"] = pd.cut(a, bins=bins, labels=labels, right=True).astype(
        pd.CategoricalDtype(categories=labels, ordered=True)
    )
    return df

def pct_and_counts(series01: pd.Series):
    valid = series01.dropna()
    n = len(valid)
    if n == 0:
        return np.nan, 0, 0
    yes = int((valid == 1).sum())
    pct = yes / n * 100.0
    return pct, yes, n

def group_summary(df: pd.DataFrame, group_cols, both_col="both_agree"):
    g = df.groupby(group_cols, observed=False)[both_col].apply(lambda s: pct_and_counts(s))
    out = pd.DataFrame(g.tolist(), index=g.index, columns=["pct_both_agree", "both_yes", "n_valid"])
    return out

def main():
    ap = build_parser(__doc__.strip().splitlines()[0], __file__)
    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)

    for col in ["country", "age", Q16_N_COL, Q16_O_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = add_age_bin(df)

    n_agree = pd.Series(agree45(df[Q16_N_COL]), index=df.index)
    o_agree = pd.Series(agree45(df[Q16_O_COL]), index=df.index)

    n_dis = pd.Series(disagree12(df[Q16_N_COL]), index=df.index)
    o_dis = pd.Series(disagree12(df[Q16_O_COL]), index=df.index)

    both_disagree = np.where(
        np.isnan(n_dis) | np.isnan(o_dis),
        np.nan,
        ((n_dis == 1) & (o_dis == 1)).astype(int)
    )
    df["both_disagree"] = both_disagree

    both = np.where(
        np.isnan(n_agree) | np.isnan(o_agree),
        np.nan,
        ((n_agree == 1) & (o_agree == 1)).astype(int)
    )
    df["both_agree"] = both

    overall_pct, overall_yes, overall_n = pct_and_counts(df["both_agree"])
    print("\nQ16 N & O overlap (Agree/Strongly agree on BOTH)")
    print(f"Overall: {overall_pct:.2f}%  ({overall_yes}/{overall_n} valid respondents)\n")

    by_country = group_summary(df, ["country"])
    print("By country (pct_both_agree, both_yes, n_valid):")
    print(by_country.sort_values("pct_both_agree", ascending=False).to_string())
    print()

    by_age = group_summary(df.dropna(subset=["age_bin"]), ["age_bin"])
    print("By age bin (pct_both_agree, both_yes, n_valid):")
    print(by_age.to_string())
    print()

    by_country_age = group_summary(df.dropna(subset=["age_bin"]), ["country", "age_bin"])
    print("By country × age bin (pct_both_agree, both_yes, n_valid):")
    print(by_country_age.sort_values("pct_both_agree", ascending=False).to_string())
    print()

    plot_country_age_grouped_bar(
        by_country_age,
        value_col="pct_both_agree",
        out_png=str(outdir / "q16_both_agree_country_age.png"),
        title="Q16 contradiction: Agree 4–5 on BOTH (N and O)",
        ylabel="% Agree on BOTH",
    )

    overall_pct_d, overall_yes_d, overall_n_d = pct_and_counts(df["both_disagree"])
    print("Q16 N & O overlap (Disagree/Strongly disagree on BOTH)")
    print(f"Overall: {overall_pct_d:.2f}%  ({overall_yes_d}/{overall_n_d} valid respondents)\n")

    by_country_d = group_summary(df, ["country"], both_col="both_disagree").rename(
        columns={"pct_both_agree": "pct_both_disagree"}
    )

    by_age_d = group_summary(df.dropna(subset=["age_bin"]), ["age_bin"], both_col="both_disagree").rename(
        columns={"pct_both_agree": "pct_both_disagree"}
    )

    by_country_age_d = group_summary(
        df.dropna(subset=["age_bin"]),
        ["country", "age_bin"],
        both_col="both_disagree"
    ).rename(
        columns={"pct_both_agree": "pct_both_disagree"}
    )

    print("By country (pct_both_disagree, both_yes, n_valid):")
    print(by_country_d.sort_values("pct_both_disagree", ascending=False).to_string())
    print()

    print("By age bin (pct_both_disagree, both_yes, n_valid):")
    print(by_age_d.to_string())
    print()

    print("By country × age bin (pct_both_disagree, both_yes, n_valid):")
    print(by_country_age_d.sort_values("pct_both_disagree", ascending=False).to_string())
    print()

    plot_country_age_grouped_bar(
        by_country_age_d,
        value_col="pct_both_disagree",
        out_png=str(outdir / "q16_both_disagree_country_age.png"),
        title="Q16 contradiction: Disagree 1–2 on BOTH (N and O)",
        ylabel="% Disagree on BOTH",
    )

    def flatten(df_in: pd.DataFrame, grouping_name: str) -> pd.DataFrame:
        out = df_in.reset_index()
        out.insert(0, "grouping", grouping_name)
        return out

    combined = pd.concat([
        flatten(by_country, "country"),
        flatten(by_age, "age_bin"),
        flatten(by_country_age, "country_age_bin"),
    ], ignore_index=True)

    save_csv(by_country.reset_index(), outdir / "q16_both_agree_by_country.csv")
    save_csv(by_age.reset_index(), outdir / "q16_both_agree_by_age.csv")
    save_csv(by_country_age.reset_index(), outdir / "q16_both_agree_by_country_age.csv")
    save_csv(by_country_d.reset_index(), outdir / "q16_both_disagree_by_country.csv")
    save_csv(by_age_d.reset_index(), outdir / "q16_both_disagree_by_age.csv")
    save_csv(by_country_age_d.reset_index(), outdir / "q16_both_disagree_by_country_age.csv")
    save_csv(pd.DataFrame([{
        "measure": "both_agree",
        "overall_pct": float(overall_pct),
        "both_yes": int(overall_yes),
        "n_valid": int(overall_n),
    }, {
        "measure": "both_disagree",
        "overall_pct": float(overall_pct_d),
        "both_yes": int(overall_yes_d),
        "n_valid": int(overall_n_d),
    }]), outdir / "q16_overlap_overall_summary.csv")
    save_csv(combined, outdir / "q16_both_agree_combined_summary.csv")
    print(f"Saved outputs to: {outdir}")

if __name__ == "__main__":
    main()
