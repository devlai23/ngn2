"""
initial testing with media cynicism and trust indices / plotting by ideology
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import build_parser, ensure_outdir, save_csv


COLS = {
    "country": "country",
    "ideology": "Q18",
    "trust_item_1": "Q16_grid_6",
    "trust_item_2": "Q16_grid_8",
    "cyn_item_1": "Q16_grid_3",
    "cyn_item_2": "Q16_grid_4",
}

DK_COMMON = {99, 98, 97}
IDEOLOGY_DK = {8, 99, 98, 97}


def to_numeric_clean(s: pd.Series, dk_codes: set) -> pd.Series:
    s = s.replace("__NA__", np.nan)
    s = pd.to_numeric(s, errors="coerce")
    return s.where(~s.isin(dk_codes), np.nan)


def summarize_by_country(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = df.groupby(COLS["country"])[value_col]
    out = g.agg(["mean", "count", "std"]).rename(columns={"count": "n"})
    out["se"] = out["std"] / np.sqrt(out["n"])
    out["ci95_lo"] = out["mean"] - 1.96 * out["se"]
    out["ci95_hi"] = out["mean"] + 1.96 * out["se"]
    return out.drop(columns=["std"])


def barh_with_ci(summary: pd.DataFrame, title: str, xlabel: str, outpath=None):
    summary = summary.sort_values("mean", ascending=True)

    y = np.arange(len(summary))
    means = summary["mean"].values
    xerr = np.vstack(
        [means - summary["ci95_lo"].values, summary["ci95_hi"].values - means]
    )

    plt.figure(figsize=(10, max(4, 0.35 * len(summary))))
    plt.barh(y, means, xerr=xerr, capsize=3)
    plt.yticks(y, summary.index.astype(str))
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


def ideology_trust_plot(df: pd.DataFrame, countries, outpath=None):
    ideos = list(range(1, 8))
    sub = df[df[COLS["country"]].isin(countries)].copy()

    grp = (
        sub.groupby([COLS["country"], COLS["ideology"]])["trust_mainstream"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    for c in countries:
        cdat = grp[grp[COLS["country"]] == c].set_index(COLS["ideology"]).reindex(ideos)
        plt.plot(ideos, cdat["trust_mainstream"].values, marker="o", label=str(c))

    plt.xticks(ideos)
    plt.xlabel("Ideology (Q18): 1=Very right ... 7=Very left")
    plt.ylabel("Mean trust_mainstream (mean of Q16_grid_6 & Q16_grid_8)")
    plt.title("Trust vs Ideology, by Country")
    plt.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()
    return grp


def violin_by_country(df, value_col, countries, title, ylabel, outpath=None):
    sub = df[df["country"].isin(countries)].dropna(subset=[value_col]).copy()

    data = [sub[sub["country"] == c][value_col].values for c in countries]

    plt.figure(figsize=(max(8, 1.1 * len(countries)), 6))
    plt.violinplot(data, showmeans=True, showmedians=True, showextrema=False)

    plt.xticks(np.arange(1, len(countries) + 1), countries, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


def main():
    ap = build_parser(__doc__.strip().splitlines()[0], __file__)
    ap.add_argument(
        "--top_countries",
        type=int,
        default=8,
        help="For ideology plot, number of largest-sample countries to include",
    )
    ap.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help="Optional explicit list of countries for ideology plot (overrides top_countries)",
    )
    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)

    for k, col in COLS.items():
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' required for '{k}'.")

    df[COLS["ideology"]] = to_numeric_clean(df[COLS["ideology"]], IDEOLOGY_DK)

    for k in ["trust_item_1", "trust_item_2", "cyn_item_1", "cyn_item_2"]:
        df[COLS[k]] = to_numeric_clean(df[COLS[k]], DK_COMMON)

    df["trust_mainstream"] = df[[COLS["trust_item_1"], COLS["trust_item_2"]]].mean(
        axis=1, skipna=True
    )
    df["cynicism"] = df[[COLS["cyn_item_1"], COLS["cyn_item_2"]]].mean(
        axis=1, skipna=True
    )

    trust_sum = summarize_by_country(
        df.dropna(subset=["trust_mainstream"]), "trust_mainstream"
    )
    cyn_sum = summarize_by_country(df.dropna(subset=["cynicism"]), "cynicism")
    save_csv(trust_sum.reset_index(), outdir / "trust_by_country_summary.csv")
    save_csv(cyn_sum.reset_index(), outdir / "cynicism_by_country_summary.csv")

    barh_with_ci(
        trust_sum,
        "Trust in mainstream news by country (mean of Q16_grid_6 & Q16_grid_8)",
        "Mean trust_mainstream (1–5)",
        outpath=os.path.join(outdir, "trust_by_country.png"),
    )

    barh_with_ci(
        cyn_sum,
        "Media cynicism by country (mean of Q16_grid_3 & Q16_grid_4)",
        "Mean cynicism (1–5)",
        outpath=os.path.join(outdir, "cynicism_by_country.png"),
    )

    if args.countries:
        countries = args.countries
    else:
        tmp = df.dropna(subset=[COLS["ideology"], "trust_mainstream"])
        countries = (
            tmp[COLS["country"]].value_counts().head(args.top_countries).index.tolist()
        )

    ideology_summary = ideology_trust_plot(
        df.dropna(subset=[COLS["ideology"], "trust_mainstream"]),
        countries=countries,
        outpath=os.path.join(outdir, "trust_vs_ideology_by_country.png"),
    )
    save_csv(ideology_summary, outdir / "trust_vs_ideology_by_country.csv")

    violin_by_country(
        df,
        "trust_mainstream",
        countries,
        "Trust distribution by country",
        "Trust (1–5)",
        outpath=os.path.join(outdir, "trust_violin.png"),
    )

    violin_by_country(
        df,
        value_col="cynicism",
        countries=countries,
        title="Cynicism distribution by country",
        ylabel="Cynicism (1–5)",
        outpath=os.path.join(outdir, "cynicism_violin_vertical.png"),
    )

    violin_input = (
        df[df["country"].isin(countries)][["country", "trust_mainstream", "cynicism"]]
        .dropna(how="all")
        .copy()
    )
    save_csv(violin_input, outdir / "country_level_index_values.csv")

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
