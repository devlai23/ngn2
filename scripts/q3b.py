"""
research question 3 part b

In some countries (namely Nigeria, Brazil, India), local news and national news are consumed (in conjunction) more 
than other countries because of linguistic differences.

Q4_A,B,E,F → “I get news often from national TV, local TV, local newspapers, national newspapers” 
Q4 Options A + F (when values are 3 for one or the other or both)
Q4 Options B + E (when values are 3 for one or the other or both)
Compare test linguistically diverse countries (nigeria, india, brazil) against others
Chi-square test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from common import build_parser, ensure_outdir, save_csv

COUNTRY_COL = "country"

DIVERSE_CODES = {"ng", "br", "in"} 
COMPARISON_CODES = {"us", "uk", "kr"} 
Q4_A = "Q4_grid_1"  # National TV
Q4_B = "Q4_grid_2"  # Local TV
Q4_E = "Q4_grid_5"  # Local newspapers/magazines
Q4_F = "Q4_grid_6"  # National newspapers/magazines

VALID_Q4 = {0, 1, 2, 3}
DK_CODE = 99

def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("__NA__", np.nan), errors="coerce")

def clean_q4(s: pd.Series) -> pd.Series:
    x = clean_numeric(s)
    x = x.where(~x.isin([DK_CODE]), np.nan)
    x = x.where(x.isin(list(VALID_Q4)), np.nan)
    return x

def make_group(country):
    if country in DIVERSE_CODES:
        return "diverse_lang"
    elif country in COMPARISON_CODES:
        return "us_uk_kr"
    else:
        return np.nan 

def main():
    parser = build_parser(__doc__.strip().splitlines()[0], __file__)
    parser.add_argument("--country_col", default=COUNTRY_COL)
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)

    tmp = df[[args.country_col, Q4_A, Q4_B, Q4_E, Q4_F]].copy()
    tmp["country_group"] = tmp[args.country_col].map(make_group)
    tmp = tmp.dropna(subset=["country_group"]).copy()

    for col in [Q4_A, Q4_B, Q4_E, Q4_F]:
        tmp[col] = clean_q4(tmp[col])

    tmp = tmp.dropna(subset=[Q4_A, Q4_B, Q4_E, Q4_F]).copy()

    tmp["local_any_often"] = ((tmp[Q4_B] == 3) | (tmp[Q4_E] == 3)).astype(int)
    tmp["national_any_often"] = ((tmp[Q4_A] == 3) | (tmp[Q4_F] == 3)).astype(int)
    tmp["both_local_national_often"] = (
        (tmp["local_any_often"] == 1) & (tmp["national_any_often"] == 1)
    ).astype(int)

    n_by_country = tmp[args.country_col].value_counts().sort_index().rename_axis("country").reset_index(name="n")
    save_csv(n_by_country, outdir / "q3b_n_by_country.csv")
    print("N by country:")
    print(tmp[args.country_col].value_counts().sort_index())

    n_by_group = tmp["country_group"].value_counts().rename_axis("country_group").reset_index(name="n")
    save_csv(n_by_group, outdir / "q3b_n_by_group.csv")
    print("\nN by grouped hypothesis bucket:")
    print(tmp["country_group"].value_counts())

    pct_country = (
        (tmp.groupby(args.country_col)["both_local_national_often"].mean() * 100)
        .sort_values(ascending=False)
        .round(1)
        .rename("pct_both_local_national_often")
        .rename_axis("country")
        .reset_index()
    )
    save_csv(pct_country, outdir / "q3b_pct_both_by_country.csv")
    print("\n% BOTH local+national (Often) by country:")
    print(pct_country.set_index("country")["pct_both_local_national_often"].to_string())

    pct_group = (
        (tmp.groupby("country_group")["both_local_national_often"].mean() * 100)
        .round(1)
        .rename("pct_both_local_national_often")
        .rename_axis("country_group")
        .reset_index()
    )
    save_csv(pct_group, outdir / "q3b_pct_both_by_group.csv")
    print("\n% BOTH local+national (Often) by group:")
    print(pct_group.set_index("country_group")["pct_both_local_national_often"].to_string())

    ct = pd.crosstab(tmp["country_group"], tmp["both_local_national_often"])
    ct = ct.reindex(index=["diverse_lang", "us_uk_kr"], columns=[0, 1], fill_value=0)
    save_csv(ct.reset_index(), outdir / "q3b_group_by_both_counts.csv")
    print("\nContingency table (counts):")
    print(ct)

    row_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    save_csv(row_pct.reset_index(), outdir / "q3b_group_by_both_rowpct.csv")
    print("\nRow % (within group):")
    print(row_pct.round(1))

    chi2, p, dof, expected = chi2_contingency(ct)
    print("\nChi-square test (group × BOTH local+national):")
    print(f"chi2 = {chi2:.4f}")
    print(f"dof  = {dof}")
    print(f"p    = {p:.4e}")

    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
    save_csv(expected_df.reset_index(), outdir / "q3b_group_by_both_expected.csv")
    print("\nExpected counts:")
    print(expected_df.round(2))

    n = ct.to_numpy().sum()
    phi = np.sqrt(chi2 / n) if n > 0 else np.nan
    p_div = row_pct.loc["diverse_lang", 1]
    p_cmp = row_pct.loc["us_uk_kr", 1]
    save_csv(pd.DataFrame([{
        "chi2": float(chi2),
        "dof": int(dof),
        "p": float(p),
        "phi": float(phi),
        "pct_point_diff_diverse_minus_comparison": float(p_div - p_cmp),
    }]), outdir / "q3b_group_by_both_test.csv")
    print(f"\nPhi (effect size) = {phi:.4f}")
    print(f"\nDifference in % BOTH (diverse_lang - us_uk_kr) = {p_div - p_cmp:.1f} percentage points")
    print(f"\nSaved outputs to: {Path(outdir)}")


if __name__ == "__main__":
    main()
