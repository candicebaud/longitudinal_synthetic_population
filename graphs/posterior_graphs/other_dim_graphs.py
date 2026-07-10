
"""
population_graphs.py

Reusable plotting and table-generation utilities for posterior / population validation graphs.

Expected workflow:
1. In a notebook, define/load your dataframe and metadata.
2. Define a mapping(df, year) function if df_type == "time_independent".
3. Call prepare_dfs_to_plot(...).
4. Call run_population_graphs_from_config(...).

The module assumes the prepared dataframes contain, when relevant:
age, gender, hh_income, employment_status, education_status,
has_license, has_ga, car_available, canton_residence, urban.

Baud Candice
Fri July 10 10:33:00 2026
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Callable, Iterable

from graphs_style import *

import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# General helpers
# ============================================================

def clean_filename(text: str) -> str:
    """Return a safe lowercase filename component."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def ensure_output_folder(output_folder) -> Path:
    """Create output folder if needed and return it as a Path."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def make_title(base_title: str, df_name_for_title: str, df_type: str, year) -> str:
    """Create a title consistent with your notebook convention."""
    if df_type == "time_independent":
        return f"{base_title} - {df_name_for_title} projected in {year}"
    if df_type == "time_dependent":
        return f"{base_title} - {df_name_for_title} {year}"
    raise ValueError("df_type must be 'time_independent' or 'time_dependent'.")


def make_filename(base_filename: str, df_name: str, year, suffix: str = "pdf") -> str:
    """Create a filename consistent with your notebook convention."""
    return f"{base_filename}_dataframe_{df_name}_year_{year}.{suffix}"


def normalize_or_create_weights(
    df: pd.DataFrame,
    weight_column: str | None,
    weight_column_name: str = "weight",
    normalize_weights: bool = True,
) -> pd.DataFrame:
    """
    Add a standardized weight column.

    If weight_column is None, unit weights are created.
    If weight_column is provided, it is copied into weight_column_name.
    If normalize_weights=True, weights are divided by their total.
    """
    df = df.copy()

    if weight_column is None:
        df[weight_column_name] = 1.0
    else:
        if weight_column not in df.columns:
            raise ValueError(f"weight_column='{weight_column}' not found in dataframe.")
        df[weight_column_name] = df[weight_column].astype(float)

    if normalize_weights:
        total = df[weight_column_name].sum()
        if total <= 0:
            raise ValueError("The sum of weights must be positive.")
        df[weight_column_name] = df[weight_column_name] / total

    return df


def prepare_dfs_to_plot(
    df: pd.DataFrame,
    df_type: str,
    *,
    df_years_to_project: Iterable[int] | None = None,
    df_year_observed: int | None = None,
    weight_column: str | None = None,
    weight_column_name: str = "weight",
    mapping: Callable[[pd.DataFrame, int], pd.DataFrame] | None = None,
    normalize_weights: bool = True,
) -> tuple[list[pd.DataFrame], list[int]]:
    """
    Prepare a list of dataframes to plot.

    For time_independent data, mapping(df, year) is applied for each projection year.
    For time_dependent data, the original dataframe is used directly.
    """
    if df_type == "time_independent":
        if df_years_to_project is None:
            raise ValueError("df_years_to_project must be provided for time_independent data.")
        if mapping is None:
            raise ValueError("mapping must be provided for time_independent data.")

        dfs_to_plot = []
        dfs_to_plot_years = []

        for year in df_years_to_project:
            df_t = mapping(df, year)
            df_t = normalize_or_create_weights(
                df_t,
                weight_column=weight_column,
                weight_column_name=weight_column_name,
                normalize_weights=normalize_weights,
            )
            dfs_to_plot.append(df_t)
            dfs_to_plot_years.append(year)

        return dfs_to_plot, dfs_to_plot_years

    if df_type == "time_dependent":
        df_t = normalize_or_create_weights(
            df,
            weight_column=weight_column,
            weight_column_name=weight_column_name,
            normalize_weights=normalize_weights,
        )
        return [df_t], [df_year_observed]

    raise ValueError(
        f"Unknown df_type: {df_type}. "
        "Expected 'time_independent' or 'time_dependent'."
    )


def weighted_average_binary(df: pd.DataFrame, binary_col: str, weight_col: str) -> float:
    """Weighted average for a binary 0/1 variable."""
    work = df[[binary_col, weight_col]].dropna().copy()
    work = work[work[weight_col] > 0]
    if len(work) == 0:
        return np.nan
    return np.average(work[binary_col].astype(float), weights=work[weight_col].astype(float))

def truncated_poisson_moments(lmbda, max_k):
    """
    Mean and variance of a Poisson(lambda) truncated to {0, ..., max_k}.
    """
    k = np.arange(0, max_k + 1)

    pmf = poisson.pmf(k, lmbda)
    normalization = poisson.cdf(max_k, lmbda)

    if normalization == 0:
        return np.nan, np.nan

    truncated_pmf = pmf / normalization

    mean = np.sum(k * truncated_pmf)
    second_moment = np.sum((k ** 2) * truncated_pmf)
    var = second_moment - mean ** 2

    return mean, var


# ============================================================
# Default labels, colors, and mappings
# ============================================================

INCOME_CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

INCOME_LABEL_MAP = {
    1: "< 2,000",
    2: "2,000-4,000",
    3: "4,000-6,000",
    4: "6,000-8,000",
    5: "8,000-10,000",
    6: "10,000-12,000",
    7: "12,000-14,000",
    8: "14,000-16,000",
    9: "> 16,000",
}

INCOME_LABEL_MAP_SHORT = {
    1: "<2k",
    2: "2k-\n4k",
    3: "4k-\n6k",
    4: "6k-\n8k",
    5: "8k-\n10k",
    6: "10k-\n12k",
    7: "12k-\n14k",
    8: "14k-\n16k",
    9: ">16k",
}

INCOME_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
    6: "#8c564b",
    7: "#e377c2",
    8: "#7f7f7f",
    9: "#bcbd22",
}

EMPLOYMENT_CATEGORIES = [1, 2, 4, 5]

EMPLOYMENT_LABEL_MAP = {
    1: "Employed",
    2: "Unemployed or not economically active",
    4: "Age < 15",
    5: "Retired",
}

EMPLOYMENT_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    4: "#d62728",
    5: "#9467bd",
}

STUDENT_CATEGORIES = [0, 1]

STUDENT_LABEL_MAP = {
    0: "Not student",
    1: "Student",
}

STUDENT_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
}

URBAN_LABEL_MAP = {
    0: "Rural",
    1: "Urban",
}

AGE_BINS_FULL = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64, 110]
AGE_LABELS_FULL = [
    "0-15", "15-20", "20-25", "25-30", "30-35", "35-40",
    "40-45", "45-50", "50-55", "55-60", "60-64", "64+",
]

AGE_BINS_INCOME = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64]
AGE_LABELS_INCOME = [
    "15-20", "20-25", "25-30", "30-35", "35-40",
    "40-45", "45-50", "50-55", "55-60", "60-64",
]

CANTON_TO_NUMBER = {
    "Zurich": 1,
    "Bern": 2,
    "Lucerne": 3,
    "Uri": 4,
    "Schwyz": 5,
    "Obwalden": 6,
    "Nidwalden": 7,
    "Glarus": 8,
    "Zug": 9,
    "Fribourg": 10,
    "Solothurn": 11,
    "Basel_Stadt": 12,
    "Basel_Landschaft": 13,
    "Schaffhausen": 14,
    "Appenzell_Ausserrhoden": 15,
    "Appenzell_Innerrhoden": 16,
    "St_Gallen": 17,
    "Graubunden": 18,
    "Aargau": 19,
    "Thurgau": 20,
    "Ticino": 21,
    "Vaud": 22,
    "Valais": 23,
    "Neuchatel": 24,
    "Geneva": 25,
    "Jura": 26,
}

NUMBER_TO_CANTON = {v: k for k, v in CANTON_TO_NUMBER.items()}

CANTON_ABBREV_CLEAN = {
    "Zurich": "ZH",
    "Bern": "BE",
    "Vaud": "VD",
    "Aargau": "AG",
    "St_Gallen": "SG",
    "Geneva": "GE",
    "Lucerne": "LU",
    "Valais": "VS",
    "Ticino": "TI",
    "Fribourg": "FR",
    "Basel_Landschaft": "BL",
    "Thurgau": "TG",
    "Solothurn": "SO",
    "Graubunden": "GR",
    "Basel_Stadt": "BS",
    "Neuchatel": "NE",
    "Schwyz": "SZ",
    "Zug": "ZG",
    "Schaffhausen": "SH",
    "Jura": "JU",
    "Appenzell_Ausserrhoden": "AR",
    "Nidwalden": "NW",
    "Glarus": "GL",
    "Obwalden": "OW",
    "Uri": "UR",
    "Appenzell_Innerrhoden": "AI",
}


def add_canton_columns(
    df: pd.DataFrame,
    canton_col: str = "canton_residence",
) -> pd.DataFrame:
    """Add canton_name and canton_abbrev columns from numeric canton code."""
    df = df.copy()
    df["canton_name"] = df[canton_col].map(NUMBER_TO_CANTON)
    df["canton_abbrev"] = df["canton_name"].map(CANTON_ABBREV_CLEAN)
    return df


# ============================================================
# Plotting helpers
# ============================================================

def hist_categorical(
    ax: plt.Axes,
    x,
    *,
    weights=None,
    normalize=True,
    rotation=90,
    tick_fontsize=6,
    alpha=0.35,
    show_values=True,
    value_fontsize=5,
    value_format="{:.1f}",
    ylim=None,
):
    """Weighted categorical histogram, optionally normalized to percentages."""
    x = np.asarray(x, dtype=object)

    mask = pd.notna(x)
    x = x[mask]

    if weights is not None:
        weights = np.asarray(weights, dtype=float)[mask]

    categories = np.unique(x)

    if weights is None:
        counts = np.array([np.sum(x == cat) for cat in categories], dtype=float)
    else:
        counts = np.array([weights[x == cat].sum() for cat in categories], dtype=float)

    order = np.argsort(-counts)
    categories = categories[order]
    counts = counts[order]

    if normalize:
        total = counts.sum()
        counts = 100 * counts / total if total > 0 else counts
        ax.set_ylabel("Weighted share (%)")
    else:
        ax.set_ylabel("Weighted count" if weights is not None else "Count")

    positions = np.arange(len(categories))
    color_cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_color = color_cycle[0]

    bars = ax.bar(
        positions,
        counts,
        edgecolor="black",
        linewidth=0.8,
        alpha=alpha,
        color=bar_color,
    )

    if show_values:
        for bar, value in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                value_format.format(value),
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
            )

    if ylim is not None:
        ax.set_ylim(ylim)
    elif len(counts) > 0:
        ax.set_ylim(0, max(counts) * 1.12)

    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=rotation, ha="right", fontsize=tick_fontsize)

    return categories, counts


def plot_stacked_share(
    table: pd.DataFrame,
    *,
    x_col: str,
    category_col: str,
    value_col: str,
    category_order: list,
    category_label_map: dict,
    category_colors: dict,
    ax: plt.Axes | None = None,
    figsize=(10, 6),
):
    """Plot a stacked bar chart from a long weighted-share table."""
    plot_table = (
        table.pivot(index=x_col, columns=category_col, values=value_col)
        .fillna(0)
        .reindex(columns=category_order, fill_value=0)
    )

    plot_table = plot_table * 100
    plot_colors = [category_colors[c] for c in category_order]
    plot_table = plot_table.rename(columns=category_label_map)

    ax = plot_table.plot(
        kind="bar",
        stacked=True,
        figsize=figsize,
        color=plot_colors,
        ax=ax,
    )

    return ax, plot_table


def weighted_share_table(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    category_col: str,
    weight_col: str,
) -> pd.DataFrame:
    """Compute weighted shares of category_col inside each group."""
    table = (
        df.groupby(group_cols + [category_col], as_index=False, observed=True)
        .agg(weighted_count=(weight_col, "sum"))
    )

    table["weighted_share"] = (
        table["weighted_count"]
        / table.groupby(group_cols)["weighted_count"].transform("sum")
    )

    return table


# ============================================================
# Individual plots / tables
# ============================================================

def plot_income_by_gender(config: dict, dfs_to_plot: list[pd.DataFrame], dfs_to_plot_years: list[int]):
    """
    Plot household income distribution for all, women, and men.

    This keeps the original style:
    - setup_style(...)
    - hist_density_weights(...)
    - styled_axes(...)
    - women dashed line
    - men solid line
    - probability density on y-axis
    """
    output_folder = ensure_output_folder(config["output_folder"])
    weight_column_name = config["weight_column_name"]

    base_filename_income_gender = "income_category_shares_by_gender"
    base_title_income_gender = "Income by gender"

    for i in range(len(dfs_to_plot_years)):

        filename_income_gender = (
            base_filename_income_gender
            + "_dataframe_"
            + config["df_name"]
            + "_year_"
            + str(dfs_to_plot_years[i])
            + ".pdf"
        )

        if config["df_type"] == "time_independent":
            title_income_gender = (
                base_title_income_gender
                + " - "
                + config["df_name_for_title"]
                + " projected in "
                + str(dfs_to_plot_years[i])
            )

        elif config["df_type"] == "time_dependent":
            title_income_gender = (
                base_title_income_gender
                + " - "
                + config["df_name_for_title"]
                + " "
                + str(dfs_to_plot_years[i])
            )

        else:
            raise ValueError("df_type must be 'time_independent' or 'time_dependent'.")

        df_i = dfs_to_plot[i].copy()
        df_i = df_i[df_i["age"] > 7]
        df_i = df_i[df_i["employment_status"] == 1]

        df_i[weight_column_name] = df_i[weight_column_name]/np.sum(df_i[weight_column_name] )

        setup_style(column="one", fontsize=8, use_tex=False)
        fig, ax = plt.subplots()

        # --- data
        x = np.array(df_i["hh_income"])
        w = np.array(df_i[weight_column_name])
        g = np.array(df_i["gender"])

        # --- overall histogram (weighted density)
        bins = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        _, bin_edges, _ = hist_density_weights(
            ax,
            x,
            weights=w,
            bins=bins,
            label="All",
            alpha=0.3,
        )

        # --- bin centers
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # --- masks
        mask_w = g == 1
        mask_m = g == 0

        # --- women density
        counts_w, _ = np.histogram(
            x[mask_w],
            bins=bin_edges,
            weights=w[mask_w],
            density=True,
        )

        # --- men density
        counts_m, _ = np.histogram(
            x[mask_m],
            bins=bin_edges,
            weights=w[mask_m],
            density=True,
        )

        # --- plot lines
        ax.plot(
            bin_centers,
            counts_w,
            linewidth=1,
            linestyle="--",
            label="Women",
        )

        ax.plot(
            bin_centers,
            counts_m,
            linewidth=1,
            linestyle="-",
            label="Men",
        )

        # --- formatting
        styled_axes(ax, xbins=5, ybins=5)

        ax.set_xlabel("Income")
        ax.set_ylabel("Probability density")
        ax.set_title(title_income_gender)
        ax.set_ylim(0, 0.5)
        ax.legend()

        # --- readable income labels
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(
            [INCOME_LABEL_MAP_SHORT[c] for c in INCOME_CATEGORIES[:-1]],
            rotation=45,
            ha="right",
        )

        plt.tight_layout()

        plt.savefig(
            output_folder / filename_income_gender,
            format="pdf",
        )

        plt.close()


def plot_income_by_age_group(config: dict, dfs_to_plot: list[pd.DataFrame], dfs_to_plot_years: list[int]):
    """
    Plot income category shares by age group.

    The plot is restricted to age > 7 and employment_status == 1, following your notebook.
    """
    output_folder = ensure_output_folder(config["output_folder"])
    weight_col = config["weight_column_name"]

    for df_i, year in zip(dfs_to_plot, dfs_to_plot_years):
        title = make_title("Income", config["df_name_for_title"], config["df_type"], year)
        filename = make_filename("income_category_shares_by_age_group", config["df_name"], year)

        work = df_i.copy()
        work = work[work["age"] > 7].copy()
       
        work["age_group"] = pd.cut(
            work["age"],
            bins=AGE_BINS_INCOME,
            labels=AGE_LABELS_INCOME,
            right=True,
            include_lowest=True,
        )
        work = work[work["employment_status"] == 1].copy()

        table = weighted_share_table(
            work,
            group_cols=["age_group"],
            category_col="hh_income",
            weight_col=weight_col,
        )

        ax, _ = plot_stacked_share(
            table,
            x_col="age_group",
            category_col="hh_income",
            value_col="weighted_share",
            category_order=INCOME_CATEGORIES,
            category_label_map=INCOME_LABEL_MAP,
            category_colors=INCOME_COLORS,
            figsize=(12, 7),
        )

        ax.set_ylabel("Weighted share (%)", fontsize=25)
        ax.set_xlabel("Age group", fontsize=25)
        ax.set_title(title, fontsize=25)
        ax.tick_params(axis="y", labelsize=20)
        ax.tick_params(axis="x", labelsize=14)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(
            title="Income category",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=20,
            title_fontsize=20,
        )

        plt.tight_layout()
        plt.savefig(output_folder / filename, format="pdf", bbox_inches="tight")
        plt.close()


def write_employment_status_table(
    config: dict,
    dfs_to_plot: list[pd.DataFrame],
    dfs_to_plot_years: list[int],
):
    output_folder = ensure_output_folder(config["output_folder"])
    weight_col = config["weight_column_name"]

    for df_i, year in zip(dfs_to_plot, dfs_to_plot_years):
        filename = f"employment_status_{config['df_name']}_year_{year}.csv"

        # Filter population: keep only people older than 7
        df_i = df_i[df_i["age"] > 7].copy()

        # Renormalise weights after filtering
        total_weight = df_i[weight_col].sum()

        if total_weight == 0:
            raise ValueError(
                f"Total weight is zero after filtering age > 7 for year {year}."
            )

        df_i[weight_col] = df_i[weight_col] / total_weight

        table = (
            df_i.groupby("employment_status", as_index=False, observed=True)[weight_col]
            .sum()
        )

        table["weighted_percentage"] = table[weight_col] * 100
        table["employment_status_label"] = table["employment_status"].map(
            EMPLOYMENT_LABEL_MAP
        )

        table.to_csv(output_folder / filename, index=False)


def write_employment_status_by_gender_table(
    config: dict,
    dfs_to_plot: list[pd.DataFrame],
    dfs_to_plot_years: list[int],
):
    output_folder = ensure_output_folder(config["output_folder"])
    weight_col = config["weight_column_name"]

    for df_i, year in zip(dfs_to_plot, dfs_to_plot_years):
        filename = f"employment_status_per_gender_{config['df_name']}_year_{year}.csv"

        # Filter population: keep only people older than 7
        df_i = df_i[df_i["age"] > 7].copy()

        # Renormalise weights after filtering
        total_weight = df_i[weight_col].sum()

        if total_weight == 0:
            raise ValueError(
                f"Total weight is zero after filtering age > 7 for year {year}."
            )

        df_i[weight_col] = df_i[weight_col] / total_weight

        table = (
            df_i
            .groupby(["gender", "employment_status"], as_index=False, observed=True)[weight_col]
            .sum()
        )

        table["weighted_percentage"] = (
            table[weight_col]
            / table.groupby("gender")[weight_col].transform("sum")
            * 100
        )

        table["employment_status_label"] = table["employment_status"].map(
            EMPLOYMENT_LABEL_MAP
        )

        table.to_csv(output_folder / filename, index=False)


def write_license_status_table(
    config: dict,
    dfs_to_plot: list[pd.DataFrame],
    dfs_to_plot_years: list[int],
):
    output_folder = ensure_output_folder(config["output_folder"])
    weight_col = config["weight_column_name"]

    rows = []

    for df_i, year in zip(dfs_to_plot, dfs_to_plot_years):
        work = df_i[df_i["age"] > 18].copy()

        # Renormalise weights after filtering age > 18
        total_weight = work[weight_col].sum()

        if total_weight == 0:
            raise ValueError(
                f"Total weight is zero after filtering age > 18 for year {year}."
            )

        work[weight_col] = work[weight_col] / total_weight

        proportion = weighted_average_binary(work, "has_license", weight_col)

        rows.append({
            "year": year,
            "proportion_has_license": proportion,
            "percentage_has_license": proportion * 100,
        })

    table = pd.DataFrame(rows)
    table.to_csv(output_folder / f"dl_status_{config['df_name']}.csv", index=False)

    return table


def write_license_status_by_gender_table(
    config: dict,
    dfs_to_plot: list[pd.DataFrame],
    dfs_to_plot_years: list[int],
):
    output_folder = ensure_output_folder(config["output_folder"])
    weight_col = config["weight_column_name"]

    rows = []

    for df_i, year in zip(dfs_to_plot, dfs_to_plot_years):
        work = df_i[df_i["age"] > 18].copy()

        # Renormalise weights after filtering age > 18
        total_weight = work[weight_col].sum()

        if total_weight == 0:
            raise ValueError(
                f"Total weight is zero after filtering age > 7 for year {year}."
            )

        work[weight_col] = work[weight_col] / total_weight

        for gender, g in work.groupby("gender", observed=True):
            proportion = weighted_average_binary(g, "has_license", weight_col)

            rows.append({
                "year": year,
                "gender": gender,
                "proportion_has_license": proportion,
                "percentage_has_license": proportion * 100,
            })

    table = pd.DataFrame(rows)
    table.to_csv(
        output_folder / f"dl_status_per_gender_{config['df_name']}.csv",
        index=False,
    )

    return table


def plot_canton_residence(config: dict, dfs_to_plot: list[pd.DataFrame], dfs_to_plot_years: list[int]):
    """
    Exact automated version of the old canton-residence plotting cell.
    Do not modify styling here unless you also modify the old notebook style.
    """
    output_folder = ensure_output_folder(config["output_folder"])
    weight_column_name = config["weight_column_name"]

    base_title_canton_res = "Canton of residence"
    base_filename_canton_res = "canton_res"

    for i in range(len(dfs_to_plot_years)):

        df_i = dfs_to_plot[i].copy()

        df_i["canton_name"] = df_i["canton_residence"].map(NUMBER_TO_CANTON)
        df_i["canton_abbrev"] = df_i["canton_name"].map(CANTON_ABBREV_CLEAN)

        if config["df_type"] == "time_independent":
            title_canton_res = (
                base_title_canton_res
                + " - "
                + config["df_name_for_title"]
                + " projected in "
                + str(dfs_to_plot_years[i])
            )
        elif config["df_type"] == "time_dependent":
            title_canton_res = (
                base_title_canton_res
                + " - "
                + config["df_name_for_title"]
                + " "
                + str(dfs_to_plot_years[i])
            )
        else:
            raise ValueError("df_type must be 'time_independent' or 'time_dependent'.")

        filename_canton_res = (
            base_filename_canton_res
            + "_"
            + config["df_name"]
            + "_year_"
            + str(dfs_to_plot_years[i])
            + ".pdf"
        )

        # EXACT old style
        setup_style(column="one", fontsize=8, use_tex=False)
        fig, ax = plt.subplots()

        hist_categorical(
            ax,
            df_i[df_i["age"] > 7]["canton_abbrev"],
            weights=df_i[df_i["age"] > 7][weight_column_name],
            value_fontsize=4,
            tick_fontsize=4,
            ylim=(0, 22),
        )

        ax.set_xlabel("Place of Residence")
        ax.set_title(title_canton_res)

        plt.tight_layout()
        plt.savefig(output_folder / filename_canton_res, format="pdf")
        plt.close()


def plot_canton_residence_urban(config: dict, dfs_to_plot: list[pd.DataFrame], dfs_to_plot_years: list[int]):
    """
    Exact automated version of the old canton-residence-with-urban-shares plotting cell.
    """
    output_folder = ensure_output_folder(config["output_folder"])
    weight_column_name = config["weight_column_name"]

    # base_title_canton_res_urban = "Canton of residence with urban shares"
    base_title_canton_res_urban = ""
    base_filename_canton_res_urban = "canton_res_urban"

    for i in range(len(dfs_to_plot_years)):

        df_i = dfs_to_plot[i].copy()

        df_i["urban_name"] = df_i["urban"].map(URBAN_LABEL_MAP)
        df_i["canton_name"] = df_i["canton_residence"].map(NUMBER_TO_CANTON)
        df_i["canton_abbrev"] = df_i["canton_name"].map(CANTON_ABBREV_CLEAN)

        if config["df_type"] == "time_independent":
            title_canton_res_urban = (
                base_title_canton_res_urban
                # + " - "
                + config["df_name_for_title"]
                + " projected in "
                + str(dfs_to_plot_years[i])
            )
        elif config["df_type"] == "time_dependent":
            title_canton_res_urban = (
                base_title_canton_res_urban
                # + " - "
                + config["df_name_for_title"]
                + " "
                + str(dfs_to_plot_years[i])
            )
        else:
            raise ValueError("df_type must be 'time_independent' or 'time_dependent'.")

        filename_canton_res_urban = (
            base_filename_canton_res_urban
            + "_"
            + config["df_name"]
            + "_year_"
            + str(dfs_to_plot_years[i])
            + ".pdf"
        )

        # EXACT old style
        setup_style(column="one", fontsize=8, use_tex=False)
        fig, ax = plt.subplots()

        hist_categorical_interaction(
            ax,
            x=df_i["canton_abbrev"],
            weights=df_i[weight_column_name],
            fill_by=df_i["urban_name"],
            normalize=True,
            ylim=(0, 22),
            value_fontsize=4,
            value_format="{:.1f}",
        )

        ax.set_title(title_canton_res_urban)

        plt.tight_layout()
        plt.savefig(output_folder / filename_canton_res_urban, format="pdf")
        plt.close()


def plot_employment_by_age_group(config: dict, dfs_to_plot: list[pd.DataFrame], dfs_to_plot_years: list[int]):
    output_folder = ensure_output_folder(config["output_folder"])
    weight_col = config["weight_column_name"]

    for df_i, year in zip(dfs_to_plot, dfs_to_plot_years):
        title = make_title("Employment category shares by age group", config["df_name_for_title"], config["df_type"], year)
        filename = make_filename("employment_category_shares_by_age_group", config["df_name"], year)

        work = df_i.copy()
        work = work[work["age"] > 7].copy()
        work["age_group"] = pd.cut(
            work["age"],
            bins=AGE_BINS_FULL,
            labels=AGE_LABELS_FULL,
            right=True,
            include_lowest=True,
        )

        table = weighted_share_table(
            work,
            group_cols=["age_group"],
            category_col="employment_status",
            weight_col=weight_col,
        )

        ax, _ = plot_stacked_share(
            table,
            x_col="age_group",
            category_col="employment_status",
            value_col="weighted_share",
            category_order=EMPLOYMENT_CATEGORIES,
            category_label_map=EMPLOYMENT_LABEL_MAP,
            category_colors=EMPLOYMENT_COLORS,
            figsize=(10, 6),
        )

        ax.set_ylabel("Weighted share (%)")
        ax.set_xlabel("Age group")
        ax.set_title(title)
        ax.legend(title="Employment status", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(output_folder / filename, format="pdf")
        plt.close()


def plot_employment_by_canton(config: dict, dfs_to_plot: list[pd.DataFrame], dfs_to_plot_years: list[int]):
    output_folder = ensure_output_folder(config["output_folder"])
    weight_col = config["weight_column_name"]

    for df_i, year in zip(dfs_to_plot, dfs_to_plot_years):
        title = make_title("Employment category shares by canton of residence", config["df_name_for_title"], config["df_type"], year)
        filename = make_filename("employment_category_shares_by_canton", config["df_name"], year)

        work = df_i.copy()
        work = work[work["age"] > 7].copy()
        work = add_canton_columns(work)

        table = weighted_share_table(
            work,
            group_cols=["canton_abbrev"],
            category_col="employment_status",
            weight_col=weight_col,
        )

        ax, _ = plot_stacked_share(
            table,
            x_col="canton_abbrev",
            category_col="employment_status",
            value_col="weighted_share",
            category_order=EMPLOYMENT_CATEGORIES,
            category_label_map=EMPLOYMENT_LABEL_MAP,
            category_colors=EMPLOYMENT_COLORS,
            figsize=(10, 6),
        )

        ax.set_ylabel("Weighted share (%)")
        ax.set_xlabel("Canton of residence")
        ax.set_title(title)
        ax.legend(title="Employment status", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_folder / filename, format="pdf")
        plt.close()


def plot_student_by_canton(config: dict, dfs_to_plot: list[pd.DataFrame], dfs_to_plot_years: list[int]):
    output_folder = ensure_output_folder(config["output_folder"])
    weight_col = config["weight_column_name"]

    for df_i, year in zip(dfs_to_plot, dfs_to_plot_years):
        title = make_title("Student shares by canton of residence", config["df_name_for_title"], config["df_type"], year)
        filename = make_filename("student_shares_by_canton", config["df_name"], year)

        work = df_i.copy()
        work = work[work["age"] > 7].copy()
        work = add_canton_columns(work)

        table = weighted_share_table(
            work,
            group_cols=["canton_abbrev"],
            category_col="education_status",
            weight_col=weight_col,
        )

        ax, _ = plot_stacked_share(
            table,
            x_col="canton_abbrev",
            category_col="education_status",
            value_col="weighted_share",
            category_order=STUDENT_CATEGORIES,
            category_label_map=STUDENT_LABEL_MAP,
            category_colors=STUDENT_COLORS,
            figsize=(10, 6),
        )

        ax.set_ylabel("Weighted share (%)")
        ax.set_xlabel("Canton of residence")
        ax.set_title(title)
        ax.legend(title="Education status", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_folder / filename, format="pdf")
        plt.close()


def export_number_of_jobs_by_lifespan(
    df,
    df_type,
    df_name,
    output_folder,
    max_k=11,
):
    """
    For time-independent event-level dataframes, compute the observed and theoretical
    number of work spells by lifespan bin and export the result as a CSV file.

    This function expects:
    - one row with event_name == "Birth" per individual,
    - an individual identifier column called "id",
    - a duration column called "main_duration",
    - work events whose event_name starts with "Work_".

    Parameters
    ----------
    df : pd.DataFrame
        Original event-level dataframe.
    df_type : str
        Either "time_independent" or "time_dependent".
    df_name : str
        Name used in the output filename.
    output_folder : pathlib.Path or str
        Folder where the CSV file is saved.
    max_k : int
        Maximum number of work spells used for the truncated Poisson moments.

    Returns
    -------
    pd.DataFrame or None
        The exported table if df_type == "time_independent", otherwise None.
    """

    if df_type != "time_independent":
        return None

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    required_columns = ["id", "event_name", "main_duration"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Cannot compute number of jobs because the following columns are missing: "
            + ", ".join(missing_columns)
        )

    df_birth = df[df["event_name"] == "Birth"].copy()

    if df_birth.empty:
        raise ValueError(
            "Cannot compute number of jobs because there are no rows with event_name == 'Birth'."
        )

    lifespan_df = df_birth[["id", "main_duration"]].rename(
        columns={"main_duration": "lifespan"}
    )

    df_work = df[
        df["event_name"]
        .fillna("")
        .astype(str)
        .str.startswith("Work_")
    ].copy()

    work_counts = (
        df_work
        .groupby("id")
        .size()
        .reset_index(name="number_of_spells")
    )

    df_individual = lifespan_df.merge(
        work_counts,
        on="id",
        how="left",
    )

    df_individual["number_of_spells"] = (
        df_individual["number_of_spells"]
        .fillna(0)
        .astype(int)
    )

    bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64, 110]

    labels = (
        ["[0,15]"]
        + [f"]{a},{b}]" for a, b in zip(bins[1:-1], bins[2:])]
    )

    df_individual["lifespan_bin"] = pd.cut(
        df_individual["lifespan"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    emp_stats = (
        df_individual
        .groupby("lifespan_bin", observed=True)["number_of_spells"]
        .agg(["mean", "var", "count"])
        .reset_index()
    )

    emp_stats = emp_stats.rename(
        columns={
            "mean": "observed_mean",
            "var": "observed_var",
            "count": "n_individuals",
        }
    )

    bin_centers = pd.DataFrame({
        "lifespan_bin": labels,
        "bin_center": [(a + b) / 2 for a, b in zip(bins[:-1], bins[1:])],
    })

    emp_stats = emp_stats.merge(
        bin_centers,
        on="lifespan_bin",
        how="left",
    )

    theo_means = []
    theo_vars = []

    for lifespan_center in emp_stats["bin_center"]:

        if lifespan_center <= 15:
            theo_means.append(0.0)
            theo_vars.append(0.0)

        else:
            lmbda = np.sqrt(min(lifespan_center, 64) - 15)
            theo_mean, theo_var = truncated_poisson_moments(
                lmbda=lmbda,
                max_k=max_k,
            )

            theo_means.append(theo_mean)
            theo_vars.append(theo_var)

    emp_stats["theo_mean"] = theo_means
    emp_stats["theo_var"] = theo_vars
    emp_stats["max_k"] = max_k

    output_path = output_folder / f"number_of_jobs_by_lifespan_{df_name}.csv"

    emp_stats.to_csv(
        output_path,
        index=False,
    )

    return emp_stats


# ============================================================
# Main launcher
# ============================================================


def write_license_status_by_age_gender_table(
    config: dict,
    dfs_to_plot: list[pd.DataFrame],
    dfs_to_plot_years: list[int],
):
    """
    Export weighted driving-license shares by age category and gender.
    """
    output_folder = ensure_output_folder(config["output_folder"])
    weight_col = config["weight_column_name"]

    age_bins = [0, 18, 25, 35, 45, 55, 70, 85, 110]
    age_labels = [
        "[0,18[", "[18,25[", "[25,35[", "[35,45[",
        "[45,55[", "[55,70[", "[70,85[", "[85,110["
    ]

    rows = []

    for df_i, year in zip(dfs_to_plot, dfs_to_plot_years):
        work = df_i.copy()
        work["age_group"] = pd.cut(
            work["age"],
            bins=age_bins,
            labels=age_labels,
            right=False,
            include_lowest=True,
        )
        work = work.dropna(subset=["age_group", "gender", "has_license", weight_col])

        for (age_group, gender), g in work.groupby(["age_group", "gender"], observed=True):
            prop = np.average(
                g["has_license"].astype(float),
                weights=g[weight_col].astype(float),
            )
            rows.append({
                "year": year,
                "age_group": age_group,
                "gender": gender,
                "weighted_share_license": prop,
                "weighted_percentage_license": 100 * prop,
                "weighted_population": g[weight_col].sum(),
                "n_observations": len(g),
            })

    table = pd.DataFrame(rows)
    table.to_csv(
        output_folder / f"dl_status_by_age_gender_{config['df_name']}.csv",
        index=False,
    )
    return table


PLOT_FUNCTIONS = {
    "income_by_gender": plot_income_by_gender,
    "income_by_age_group": plot_income_by_age_group,
    "employment_status_table": write_employment_status_table,
    "employment_status_by_gender_table": write_employment_status_by_gender_table,
    "license_status_table": write_license_status_table,
    "license_status_by_gender_table": write_license_status_by_gender_table,
    "license_status_by_age_gender_table": write_license_status_by_age_gender_table,
    "canton_residence": plot_canton_residence,
    "canton_residence_urban": plot_canton_residence_urban,
    "employment_by_age_group": plot_employment_by_age_group,
    "employment_by_canton": plot_employment_by_canton,
    "student_by_canton": plot_student_by_canton,
}

DEFAULT_PLOTS_TO_RUN = list(PLOT_FUNCTIONS.keys())

def run_population_graphs_from_config(
    dfs_to_plot: list[pd.DataFrame],
    config: dict,
    *,
    dfs_to_plot_years: list[int] | None = None,
    df: pd.DataFrame | None = None,
):
    """
    Run selected graph/table functions.

    Required config keys:
    - df_name
    - df_name_for_title
    - df_type
    - output_folder
    - weight_column_name

    Optional config keys:
    - dfs_to_plot_years
    - plots_to_run: list of plot/table names or "all"
    - max_number_of_jobs
    """
    config = dict(config)
    output_folder = ensure_output_folder(config["output_folder"])

    if dfs_to_plot_years is None:
        dfs_to_plot_years = config.get("dfs_to_plot_years")

    if dfs_to_plot_years is None:
        raise ValueError("dfs_to_plot_years must be provided either as an argument or in config.")

    plots_to_run = config.get("plots_to_run", DEFAULT_PLOTS_TO_RUN)

    if plots_to_run == "all":
        plots_to_run = DEFAULT_PLOTS_TO_RUN

    # ------------------------------------------------------------
    # Special outputs that do not use the standard plotting signature
    # ------------------------------------------------------------
    special_outputs = {
        "number_of_jobs_by_lifespan",
    }

    standard_plots_to_run = [
        p for p in plots_to_run
        if p not in special_outputs
    ]

    special_plots_to_run = [
        p for p in plots_to_run
        if p in special_outputs
    ]

    # ------------------------------------------------------------
    # Check standard plot names
    # ------------------------------------------------------------
    unknown_standard = [
        p for p in standard_plots_to_run
        if p not in PLOT_FUNCTIONS
    ]

    if unknown_standard:
        raise ValueError(
            "Unknown plots requested: "
            + ", ".join(unknown_standard)
            + ". Available standard plots are: "
            + ", ".join(PLOT_FUNCTIONS.keys())
            + ". Available special outputs are: "
            + ", ".join(special_outputs)
        )

    results = {}

    # ------------------------------------------------------------
    # Run standard graph/table functions
    # ------------------------------------------------------------
    for plot_name in standard_plots_to_run:
        result = PLOT_FUNCTIONS[plot_name](
            config,
            dfs_to_plot,
            dfs_to_plot_years,
        )

        if result is not None:
            results[plot_name] = result

    # ------------------------------------------------------------
    # Run special output: number of jobs by lifespan
    # ------------------------------------------------------------
    if "number_of_jobs_by_lifespan" in special_plots_to_run:

        if df is None:
            raise ValueError(
                "To run 'number_of_jobs_by_lifespan', you must pass the original "
                "event-level dataframe using df=df when calling run_population_graphs_from_config."
            )

        results["number_of_jobs_by_lifespan"] = export_number_of_jobs_by_lifespan(
            df=df,
            df_type=config["df_type"],
            df_name=config["df_name"],
            output_folder=output_folder,
            max_k=config.get("max_number_of_jobs", 11),
        )

    return {
        "output_folder": str(output_folder),
        "plots_run": list(plots_to_run),
        "results": results,
    }