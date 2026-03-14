"""
doing initial analyses, visually by generating heatmaps
by age, country to country, country x age, and total
for questions 5, 7, 8, 10, 12, 16
all heatmaps exported to knight lab quant team google drive
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import build_parser, ensure_outdir, save_csv

Q5_LABELS = {
    1: "LinkedIn",
    2: "X/Twitter",
    3: "Instagram",
    4: "Reddit",
    5: "Pinterest",
    6: "WhatsApp",
    7: "Snapchat",
    8: "Telegram",
    9: "WeChat",
    10: "Signal",
    11: "Facebook",
    12: "BlueSky",
    13: "Rumble",
    14: "TikTok",
    15: "Discord",
    16: "ChatGPT",
    17: "Perplexity",
    18: "Gemini",
    19: "Threads",
    20: "Google Search",
    21: "Bing Search",
    22: "Substack",
    23: "Google News",
    24: "Apple News",
    25: "Yahoo News",
    26: "MSN",
    27: "Spotify Podcasts",
    28: "Apple Podcasts",
    29: "YouTube",
    30: "Facebook Messenger",
    31: "Quora",
    32: "Kuaishou",
    33: "Nextdoor",
    34: "Sharechat",
    35: "InShorts",
    36: "Moj",
    37: "KaKaoTalk",
    38: "Naver",
    39: "Daum",

}

Q6_TARGETS = {
    "Q6 D: routinely fact-check": "Q6_grid_4",
    "Q6 M: seek multiple perspectives": "Q6_grid_13",
    "Q6 N: seek commentators I trust": "Q6_grid_14",
    "Q6 Q: value human > AI news": "Q6_grid_17",
    "Q6 R: more comfy asking AI than person": "Q6_grid_18",
    "Q6 S: see a lot of AI-generated content": "Q6_grid_19",
}

Q7_LABELS = {
    1: "Matters to work",
    2: "Affects someone you care about",
    3: "Affects your community",
    4: "Controversial",
    5: "Not sure if true",
    6: "Topic interests you",
    7: "Recognize creator/publisher",
    8: "Other",
    99: "Don't know",
}

Q8_LABELS = {
    1: "Delete apps",
    2: "Turn off notifications",
    3: "Check news sites less",
    4: "Change mix in feeds",
    5: "Unsubscribe newsletters/podcasts",
    6: "Take a break from device",
    7: "Talk to friend/colleague",
    8: "Spend less time scrolling",
    9: "Other",
    99: "Don't know",
}

Q12_LABELS = {
    1: "Ask someone you know",
    2: "Use a search engine",
    3: "Use an AI tool",
    4: "Visit a news site/video",
    5: "Look at comments",
    6: "See what's trending",
    7: "Other",
    8: "Do nothing",
    9: "Don't know",
}

Q16_ITEMS = {
    1: "A impact my life",
    2: "B issues I care about",
    3: "C profit over truth",
    4: "D written to manipulate",
    5: "E interested in politics",
    6: "F mainstream accurate",
    7: "G biased against my views",
    8: "H trust mainstream",
    9: "I confident fact-check",
    10: "J distinguish fact/opinion",
    11: "K determine trustworthy",
    12: "L overwhelmed",
    13: "M too stressful",
    14: "N excited new ways",
    15: "O excited traditional ways",
    16: "P hopeful view",
}

Q17_ITEMS = {
    1: "A news in general",
    2: "B mainstream news media",
    3: "C politics",
    4: "D economy",
    5: "E celebrities",
    6: "F Russia–Ukraine",
    7: "G US Pres. Trump",
    8: "H global health outbreaks",
    9: "I when watching TV",
    10: "J when online",
    11: "K when on phone",
    12: "L on social media",
    13: "M talking w/ friends/family",
}

LIKERT_1_5 = {
    1: "Strongly disagree",
    2: "Disagree",
    3: "Neutral",
    4: "Agree",
    5: "Strongly agree",
}


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("__NA__", np.nan), errors="coerce")

def agree_45(s: pd.Series) -> pd.Series:
    x = clean_numeric(s)
    x = x.where(~x.isin([99]), np.nan) 
    return ((x >= 4) & (x <= 5)).astype(float)


def selected_binary(series: pd.Series) -> pd.Series:
    s = clean_numeric(series)
    return (s == 1).astype(int)


def age_bins(df: pd.DataFrame, col="age") -> pd.Series:
    a = clean_numeric(df[col])
    bins = [18, 30, 50, 120]
    labels = ["18-29", "30-49", "50+"]
    return pd.cut(a, bins=bins, labels=labels, right=True)


def make_matrix_percent(df: pd.DataFrame, group_rows, option_cols) -> pd.DataFrame:
    mat = {}
    for opt_name, col in option_cols.items():
        mat[opt_name] = df.groupby(group_rows)[col].mean() * 100.0
    out = pd.DataFrame(mat)
    return out


def plot_overall_bar(df: pd.DataFrame, option_cols, title, outpath=None, top_n=None):
    vals = {name: df[col].mean() * 100.0 for name, col in option_cols.items()}
    s = pd.Series(vals).sort_values(ascending=True)

    if top_n is not None:
        s = s.tail(top_n)

    plt.figure(figsize=(10, max(5, 0.35 * len(s))))
    plt.barh(s.index, s.values)
    plt.xlabel("% selected")
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_heatmap(mat: pd.DataFrame, title, outpath=None, x_rotation=45):
    display = mat.copy()

    data = display.values.astype(float)
    masked = np.ma.masked_invalid(data) 

    plt.figure(figsize=(max(10, 0.6 * display.shape[1]), max(5, 0.35 * display.shape[0])))
    plt.imshow(masked, aspect="auto")
    plt.colorbar(label="% selected")

    if isinstance(display.index, pd.MultiIndex):
        ylabels = [" | ".join(map(str, tup)) for tup in display.index.to_list()]
    else:
        ylabels = display.index.astype(str).to_list()

    plt.yticks(np.arange(display.shape[0]), ylabels)
    plt.xticks(np.arange(display.shape[1]), display.columns.astype(str), rotation=x_rotation, ha="right")

    plt.title(title + " (blank = not asked)")
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


def build_option_cols(df: pd.DataFrame, prefix: str, labels: dict):
    out = {}
    for k, label in labels.items():
        col = f"{prefix}{k}"
        if col in df.columns:
            bin_col = f"{col}__sel"
            df[bin_col] = selected_binary(df[col])
            out[label] = bin_col
    return out

def dist_table_percent(df: pd.DataFrame, group_rows, cat_col: str, labels: dict) -> pd.DataFrame:
    tmp = df.copy()
    tmp[cat_col] = clean_numeric(tmp[cat_col])

    counts = (
        tmp.dropna(subset=[cat_col])
           .groupby(group_rows + [cat_col], observed=False)
           .size()
           .unstack(fill_value=0)
    )

    order = list(labels.keys())
    for k in order:
        if k not in counts.columns:
            counts[k] = 0
    counts = counts[order]

    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    pct.columns = [labels[k] for k in pct.columns]
    return pct

def plot_q12_overall_bar(df: pd.DataFrame, cat_col: str, labels: dict, title: str, outpath=None):
    s = clean_numeric(df[cat_col]).dropna()
    counts = s.value_counts().reindex(labels.keys(), fill_value=0)
    pct = counts / counts.sum() * 100.0
    pct.index = [labels[k] for k in pct.index]

    pct = pct.sort_values(ascending=True)
    plt.figure(figsize=(10, max(5, 0.35 * len(pct))))
    plt.barh(pct.index, pct.values)
    plt.xlabel("% of respondents")
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()
        
def likert_grid_dist_percent(df: pd.DataFrame, group_rows, grid_prefix: str, items: dict,
                             valid_vals=(1,2,3,4,5), dk_vals=(99,)) -> pd.DataFrame:
    frames = []
    for i, item_label in items.items():
        col = f"{grid_prefix}{i}"
        if col not in df.columns:
            continue

        tmp = df.copy()
        x = clean_numeric(tmp[col])
        x = x.where(~x.isin(dk_vals), np.nan)
        x = x.where(x.isin(valid_vals), np.nan)
        tmp["_resp"] = x
        tmp["_item"] = item_label

        frames.append(tmp[group_rows + ["_item", "_resp"]])

    long = pd.concat(frames, ignore_index=True).dropna(subset=["_resp"])

    counts = (
        long.groupby(group_rows + ["_item", "_resp"], observed=False)
            .size()
            .unstack(fill_value=0)
    )

    for v in valid_vals:
        if v not in counts.columns:
            counts[v] = 0
    counts = counts[list(valid_vals)]

    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    pct.columns = [LIKERT_1_5[v] for v in valid_vals]
    return pct

def q6_heatmap_agree_by_country(df: pd.DataFrame, outpath=None):
    tmp = df.copy()

    for label, col in Q6_TARGETS.items():
        if col not in tmp.columns:
            raise ValueError(f"Missing column {col} in CSV")
        tmp[label] = agree_45(tmp[col])

    mat = tmp.groupby("country")[list(Q6_TARGETS.keys())].mean() * 100.0

    display = mat.sort_index()
    data = display.values.astype(float)

    plt.figure(figsize=(10, max(4, 0.5 * display.shape[0])))
    plt.imshow(data, aspect="auto")
    plt.colorbar(label="% Agree (4–5)")
    plt.yticks(np.arange(display.shape[0]), display.index.astype(str))
    plt.xticks(np.arange(display.shape[1]), display.columns, rotation=30, ha="right")
    plt.title("Q6 selected items: % Agree/Strongly agree by country")
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


def avoid_binary(s: pd.Series, mode="often_or_sometimes") -> pd.Series:
    x = clean_numeric(s)
    x = x.where(~x.isin([99]), np.nan)  

    if mode == "often_or_sometimes":
        return ((x >= 2) & (x <= 3)).astype(float)
    elif mode == "often_only":
        return (x == 3).astype(float)
    else:
        raise ValueError("mode must be 'often_or_sometimes' or 'often_only'")

def plot_q17_avoid_heatmap_by_country(df: pd.DataFrame, mode="often_or_sometimes", outpath=None):
    tmp = df.copy()

    cols = []
    labels = []
    for i, item_label in Q17_ITEMS.items():
        col = f"Q17_grid_{i}"
        if col not in tmp.columns:
            raise ValueError(f"Missing column {col}")
        bin_col = f"{col}__avoid"
        tmp[bin_col] = avoid_binary(tmp[col], mode=mode)
        cols.append(bin_col)
        labels.append(item_label)

    mat = tmp.groupby("country")[cols].mean() * 100.0
    mat.columns = labels
    mat = mat.sort_index()

    data = mat.values.astype(float)
    plt.figure(figsize=(12, max(4, 0.5 * mat.shape[0])))
    plt.imshow(data, aspect="auto")
    plt.colorbar(label="% avoiding")

    plt.yticks(np.arange(mat.shape[0]), mat.index.astype(str))
    plt.xticks(np.arange(mat.shape[1]), mat.columns, rotation=30, ha="right")

    title_mode = "% Often+Sometimes (2–3)" if mode == "often_or_sometimes" else "% Often (3)"
    plt.title(f"Q17 news avoidance by country — {title_mode}")
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


def main():
    ap = build_parser(__doc__.strip().splitlines()[0], __file__)
    ap.add_argument("--top_countries", type=int, default=8,
                    help="Limit country plots to top N countries by sample size for readability")
    ap.add_argument("--top_options", type=int, default=None,
                    help="Optional: show only top N options in overall bar charts")
    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)

    for col in ["country", "age"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV.")

    df["age_bin"] = age_bins(df, "age")

    q5_cols = build_option_cols(df, "Q5_m_", Q5_LABELS)
    q7_cols = build_option_cols(df, "Q7_m_", Q7_LABELS)
    q8_cols = build_option_cols(df, "Q8_m_", Q8_LABELS)

    top_countries = df["country"].value_counts().head(args.top_countries).index.tolist()
    df_top = df[df["country"].isin(top_countries)].copy()

    plot_overall_bar(df, q5_cols, "Q5: Platforms used in last week (overall)",
                     outpath=os.path.join(outdir, "Q5_overall_bar.png"),
                     top_n=args.top_options)
    save_csv(make_matrix_percent(df.assign(overall="overall"), "overall", q5_cols).reset_index(), outdir / "Q5_overall_percent.csv")

    q5_country = make_matrix_percent(df_top, "country", q5_cols)
    save_csv(q5_country.reset_index(), outdir / "Q5_country_heatmap.csv")
    
    asked_in = {
        "Kuaishou": ["br"],
        "Nextdoor": ["us"],
        "Sharechat": ["in"],
        "InShorts": ["in"],
        "Moj": ["in"],
        "KaKaoTalk": ["kr"],
        "Naver": ["kr"],
        "Daum": ["kr"],
    }

    for col, allowed in asked_in.items():
        if col in q5_country.columns:
            q5_country.loc[~q5_country.index.isin(allowed), col] = np.nan

    plot_heatmap(q5_country, "Q5: % selected by country ",
                 outpath=os.path.join(outdir, "Q5_country_heatmap.png"))

    q5_age = make_matrix_percent(df.dropna(subset=["age_bin"]), "age_bin", q5_cols)
    save_csv(q5_age.reset_index(), outdir / "Q5_age_heatmap.csv")
    plot_heatmap(q5_age, "Q5: % selected by age bin",
                 outpath=os.path.join(outdir, "Q5_age_heatmap.png"))

    q5_country_age = make_matrix_percent(df_top.dropna(subset=["age_bin"]), ["country", "age_bin"], q5_cols)
    save_csv(q5_country_age.reset_index(), outdir / "Q5_country_age_heatmap.csv")

    for col, allowed in asked_in.items():
        if col in q5_country_age.columns:
            idx_country = q5_country_age.index.get_level_values(0)
            q5_country_age.loc[~idx_country.isin(allowed), col] = np.nan


    plot_overall_bar(df, q7_cols, "Q7: Reasons news makes you want to learn more (overall)",
                     outpath=os.path.join(outdir, "Q7_overall_bar.png"),
                     top_n=args.top_options)
    save_csv(make_matrix_percent(df.assign(overall="overall"), "overall", q7_cols).reset_index(), outdir / "Q7_overall_percent.csv")

    q7_country = make_matrix_percent(df_top, "country", q7_cols)
    save_csv(q7_country.reset_index(), outdir / "Q7_country_heatmap.csv")
    plot_heatmap(q7_country, "Q7: % selected by country ",
                 outpath=os.path.join(outdir, "Q7_country_heatmap.png"))

    q7_age = make_matrix_percent(df.dropna(subset=["age_bin"]), "age_bin", q7_cols)
    save_csv(q7_age.reset_index(), outdir / "Q7_age_heatmap.csv")
    plot_heatmap(q7_age, "Q7: % selected by age bin",
                 outpath=os.path.join(outdir, "Q7_age_heatmap.png"))

    q7_country_age = make_matrix_percent(df_top.dropna(subset=["age_bin"]), ["country", "age_bin"], q7_cols)
    save_csv(q7_country_age.reset_index(), outdir / "Q7_country_age_heatmap.csv")
    plot_heatmap(q7_country_age, "Q7: % selected by country × age bin ",
                 outpath=os.path.join(outdir, "Q7_country_age_heatmap.png"))

    plot_overall_bar(df, q8_cols, "Q8: Strategies to manage overwhelm (overall)",
                     outpath=os.path.join(outdir, "Q8_overall_bar.png"),
                     top_n=args.top_options)
    save_csv(make_matrix_percent(df.assign(overall="overall"), "overall", q8_cols).reset_index(), outdir / "Q8_overall_percent.csv")

    q8_country = make_matrix_percent(df_top, "country", q8_cols)
    save_csv(q8_country.reset_index(), outdir / "Q8_country_heatmap.csv")
    plot_heatmap(q8_country, "Q8: % selected by country ",
                 outpath=os.path.join(outdir, "Q8_country_heatmap.png"))

    q8_age = make_matrix_percent(df.dropna(subset=["age_bin"]), "age_bin", q8_cols)
    save_csv(q8_age.reset_index(), outdir / "Q8_age_heatmap.csv")
    plot_heatmap(q8_age, "Q8: % selected by age bin",
                 outpath=os.path.join(outdir, "Q8_age_heatmap.png"))

    q8_country_age = make_matrix_percent(df_top.dropna(subset=["age_bin"]), ["country", "age_bin"], q8_cols)
    save_csv(q8_country_age.reset_index(), outdir / "Q8_country_age_heatmap.csv")
    plot_heatmap(q8_country_age, "Q8: % selected by country × age bin ",
                 outpath=os.path.join(outdir, "Q8_country_age_heatmap.png"))
    
    Q12_COL = "Q12"

    plot_q12_overall_bar(
        df, Q12_COL, Q12_LABELS,
        title="Q12: What would you do to learn more? (overall)",
        outpath=os.path.join(outdir, "Q12_overall_bar.png")
    )
    save_csv(dist_table_percent(df.assign(overall="overall"), ["overall"], Q12_COL, Q12_LABELS).reset_index(), outdir / "Q12_overall_distribution.csv")

    q12_country = dist_table_percent(df_top, ["country"], Q12_COL, Q12_LABELS)
    save_csv(q12_country.reset_index(), outdir / "Q12_country_heatmap.csv")
    plot_heatmap(
        q12_country,
        title="Q12 distribution by country (row %)",
        outpath=os.path.join(outdir, "Q12_country_heatmap.png"),
        x_rotation=45
    )

    q12_age = dist_table_percent(df.dropna(subset=["age_bin"]), ["age_bin"], Q12_COL, Q12_LABELS)
    save_csv(q12_age.reset_index(), outdir / "Q12_age_heatmap.csv")
    plot_heatmap(
        q12_age,
        title="Q12 distribution by age bin (row %)",
        outpath=os.path.join(outdir, "Q12_age_heatmap.png"),
        x_rotation=45
    )

    q12_country_age = dist_table_percent(df_top.dropna(subset=["age_bin"]), ["country", "age_bin"], Q12_COL, Q12_LABELS)
    save_csv(q12_country_age.reset_index(), outdir / "Q12_country_age_heatmap.csv")
    plot_heatmap(
        q12_country_age,
        title="Q12 distribution by country × age bin (row %)",
        outpath=os.path.join(outdir, "Q12_country_age_heatmap.png"),
        x_rotation=45
    )
    
    Q16_PREFIX = "Q16_grid_"

    df_all = df.copy()
    df_all["ALL"] = "ALL"
    q16_overall = likert_grid_dist_percent(df_all, ["ALL"], Q16_PREFIX, Q16_ITEMS)
    save_csv(q16_overall.reset_index(), outdir / "Q16_overall_heatmap.csv")
    plot_heatmap(
        q16_overall,
        title="Q16 overall: % response distribution per item (rows=item, cols=Likert)",
        outpath=os.path.join(outdir, "Q16_overall_heatmap.png"),
        x_rotation=30
    )

    q16_country = likert_grid_dist_percent(df_top, ["country"], Q16_PREFIX, Q16_ITEMS)
    save_csv(q16_country.reset_index(), outdir / "Q16_country_heatmap.csv")
    plot_heatmap(
        q16_country,
        title="Q16 by country: % response distribution per item (rows=country|item)",
        outpath=os.path.join(outdir, "Q16_country_heatmap.png"),
        x_rotation=30
    )

    q16_age = likert_grid_dist_percent(df.dropna(subset=["age_bin"]), ["age_bin"], Q16_PREFIX, Q16_ITEMS)
    save_csv(q16_age.reset_index(), outdir / "Q16_age_heatmap.csv")
    plot_heatmap(
        q16_age,
        title="Q16 by age bin: % response distribution per item (rows=age_bin|item)",
        outpath=os.path.join(outdir, "Q16_age_heatmap.png"),
        x_rotation=30
    )

    q16_country_age = likert_grid_dist_percent(
        df_top.dropna(subset=["age_bin"]),
        ["country", "age_bin"],
        Q16_PREFIX,
        Q16_ITEMS
    )
    save_csv(q16_country_age.reset_index(), outdir / "Q16_country_age_heatmap.csv")
    plot_heatmap(
        q16_country_age,
        title="Q16 by country × age bin: % response distribution per item (rows=country|age_bin|item)",
        outpath=os.path.join(outdir, "Q16_country_age_heatmap.png"),
        x_rotation=30
    )

    q6_df = df.copy()
    for label, col in Q6_TARGETS.items():
        q6_df[label] = agree_45(q6_df[col])
    q6_country = q6_df.groupby("country")[list(Q6_TARGETS.keys())].mean() * 100.0
    save_csv(q6_country.reset_index(), outdir / "Q6_selected_country_heatmap.csv")
    q6_heatmap_agree_by_country(df, outpath=os.path.join(outdir, "Q6_selected_country_heatmap.png"))

    q17_df = df.copy()
    q17_cols = []
    q17_labels = []
    for i, item_label in Q17_ITEMS.items():
        col = f"Q17_grid_{i}"
        bin_col = f"{col}__avoid"
        q17_df[bin_col] = avoid_binary(q17_df[col], mode="often_or_sometimes")
        q17_cols.append(bin_col)
        q17_labels.append(item_label)
    q17_country = q17_df.groupby("country")[q17_cols].mean() * 100.0
    q17_country.columns = q17_labels
    save_csv(q17_country.reset_index(), outdir / "Q17_country_avoid_heatmap.csv")
    plot_q17_avoid_heatmap_by_country(df, mode="often_or_sometimes", outpath=os.path.join(outdir, "Q17_country_avoid_heatmap.png"))


    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
