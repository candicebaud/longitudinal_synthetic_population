"""
Reusable plotting functions for existence / birth graphs.

This module expects the same helper functions used in the notebook:
    - setup_style
    - hist_density_weights
    - styled_axes
from graphs_style.py.

Typical use from a notebook:
    from existence_graphs import run_existence_graphs
    run_existence_graphs(...)

Baud Candice
Fri July 10 10:33:00 2026
"""

from pathlib import Path

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from graphs_style import *


def _normalise_weights(df, weight_column=None, weight_column_name="weight"):
    """
    Return a copy of df with a normalized weight column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    weight_column : str or None
        Existing column containing weights. If None, all rows receive equal weight.
    weight_column_name : str
        Name of the standardized weight column used internally by the plots.
    """
    df_ = df.copy()

    if len(df_) == 0:
        df_[weight_column_name] = []
        return df_

    if weight_column is None:
        df_[weight_column_name] = 1.0
    else:
        if weight_column not in df_.columns:
            raise ValueError(f"weight_column='{weight_column}' was not found in the dataframe.")
        df_[weight_column_name] = df_[weight_column].astype(float)

    total_weight = df_[weight_column_name].sum()
    if total_weight <= 0:
        raise ValueError("The total weight must be strictly positive.")

    df_[weight_column_name] = df_[weight_column_name] / total_weight
    return df_


def prepare_dfs_to_plot(
    df,
    df_type,
    df_years_to_project=None,
    df_year_observed=None,
    mapping=None,
    weight_column=None,
    weight_column_name="weight",
):
    """
    Prepare time-independent and time-dependent dataframes for plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe loaded by the user.
    df_type : {"time_independent", "time_dependent"}
        Type of the dataframe.
    df_years_to_project : list[int] or None
        Years for which a time-independent dataframe must be projected.
    df_year_observed : int or None
        Year represented by a time-dependent dataframe.
    mapping : callable or None
        Function mapping(df_time_indep, t) -> df_time_dep.
        Required when df_type == "time_independent".
    weight_column : str or None
        Existing weight column to use. If None, equal weights are used.
    weight_column_name : str
        Name of the standardized normalized weight column used by plots.

    Returns
    -------
    tuple
        dfs_to_plot_time_indep, dfs_to_plot_time_dep, dfs_to_plot_years_time_dep
    """
    if df_type == "time_independent":
        if df_years_to_project is None:
            raise ValueError("df_years_to_project must be provided for time_independent data.")
        if mapping is None:
            raise ValueError("mapping must be provided for time_independent data.")

        dfs_to_plot_time_indep = [
            _normalise_weights(
                df,
                weight_column=weight_column,
                weight_column_name=weight_column_name,
            )
        ]

        dfs_to_plot_time_dep = []
        dfs_to_plot_years_time_dep = []

        for t in df_years_to_project:
            df_t = mapping(df, t)
            df_t = _normalise_weights(
                df_t,
                weight_column=weight_column,
                weight_column_name=weight_column_name,
            )
            dfs_to_plot_time_dep.append(df_t)
            dfs_to_plot_years_time_dep.append(t)

    elif df_type == "time_dependent":
        df_t = _normalise_weights(
            df,
            weight_column=weight_column,
            weight_column_name=weight_column_name,
        )

        dfs_to_plot_time_dep = [df_t]
        dfs_to_plot_years_time_dep = [df_year_observed]
        dfs_to_plot_time_indep = []

    else:
        raise ValueError(
            f"Unknown df_type: {df_type}. Expected 'time_independent' or 'time_dependent'."
        )

    return dfs_to_plot_time_indep, dfs_to_plot_time_dep, dfs_to_plot_years_time_dep


def _density_by_group(x, weights, mask, bin_edges):
    """
    Compute a weighted density for one group.

    If the group is empty, return zeros instead of producing warnings / NaNs.
    """
    if np.sum(mask) == 0:
        return np.zeros(len(bin_edges) - 1)

    x_group = x[mask]
    w_group = weights[mask]

    valid = np.isfinite(x_group) & np.isfinite(w_group)
    if np.sum(valid) == 0 or np.sum(w_group[valid]) <= 0:
        return np.zeros(len(bin_edges) - 1)

    counts, _ = np.histogram(
        x_group[valid],
        bins=bin_edges,
        weights=w_group[valid],
        density=True,
    )
    return counts


def plot_age_gender(
    dfs_to_plot_time_dep,
    dfs_to_plot_years_time_dep,
    df_name,
    df_name_for_title,
    output_folder,
    weight_column_name="weight",
    bins_birth=30,
    base_title_age_gender="Age distribution",
    base_filename_age_gender="age_gender",
    female_value=1,
    male_value=0,
    age_column="age",
    gender_column="gender",
    xlim=(-5, 110),
    ylim=(0, 0.025),
    show=False,
):
    """Plot age distribution by gender for each projected / observed year."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for df_i, year in zip(dfs_to_plot_time_dep, dfs_to_plot_years_time_dep):
        df_i = df_i.copy()

        required_columns = [age_column, gender_column, weight_column_name]
        missing_columns = [col for col in required_columns if col not in df_i.columns]
        if missing_columns:
            raise ValueError(f"Missing columns for age-gender plot: {missing_columns}")

        if year is None:
            title = f"{base_title_age_gender} - {df_name_for_title}"
            filename = f"{base_filename_age_gender}_{df_name}"
        else:
            title = f"{base_title_age_gender} - {df_name_for_title} projected in {year}"
            filename = f"{base_filename_age_gender}_{df_name}_year_{year}"

        setup_style(column="one", fontsize=8, use_tex=False)
        fig, ax = plt.subplots()

        x = np.asarray(df_i[age_column], dtype=float)
        w = np.asarray(df_i[weight_column_name], dtype=float)
        g = np.asarray(df_i[gender_column])

        valid = np.isfinite(x) & np.isfinite(w)
        if np.sum(valid) == 0:
            plt.close(fig)
            raise ValueError(f"No valid age values available for year {year}.")

        _, bin_edges, _ = hist_density_weights(
            ax,
            x[valid],
            weights=w[valid],
            bins=bins_birth,
            label="All",
            alpha=0.3,
        )

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        mask_women = (g == female_value) & valid
        mask_men = (g == male_value) & valid

        counts_women = _density_by_group(x, w, mask_women, bin_edges)
        counts_men = _density_by_group(x, w, mask_men, bin_edges)

        ax.plot(bin_centers, counts_women, linewidth=1, linestyle="--", label="Women")
        ax.plot(bin_centers, counts_men, linewidth=1, linestyle="-", label="Men")

        styled_axes(ax, xbins=5, ybins=5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Age")
        ax.set_ylabel("Probability density")
        ax.set_title(title)
        ax.legend()

        fig.tight_layout()
        output_path = output_folder / f"{filename}.pdf"
        fig.savefig(output_path)
        saved_files.append(output_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved_files


def plot_dob_gender(
    dfs_to_plot_time_indep,
    df_name,
    df_name_for_title,
    output_folder,
    weight_column_name="weight",
    bins_birth=30,
    base_title_dob_gender="Date of birth",
    base_filename_dob_gender="dob_gender",
    female_value=1,
    male_value=0,
    birth_date_column="main_start_date",
    gender_column="Sex",
    min_birth_year=1900,
    ylim=(0, 0.01),
    show=False,
):
    """Plot date-of-birth distribution by gender for time-independent dataframes."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for df_i in dfs_to_plot_time_indep:
        df_i = df_i.copy()

        required_columns = [birth_date_column, gender_column, weight_column_name]
        missing_columns = [col for col in required_columns if col not in df_i.columns]
        if missing_columns:
            raise ValueError(f"Missing columns for date-of-birth plot: {missing_columns}")

        df_i = df_i[df_i[birth_date_column] >= min_birth_year].copy()

        title = f"{base_title_dob_gender} - {df_name_for_title}"
        filename = f"{base_filename_dob_gender}_{df_name}"

        setup_style(column="one", fontsize=8, use_tex=False)
        fig, ax = plt.subplots()

        x = np.asarray(df_i[birth_date_column], dtype=float)
        w = np.asarray(df_i[weight_column_name], dtype=float)
        g = np.asarray(df_i[gender_column])

        valid = np.isfinite(x) & np.isfinite(w)
        if np.sum(valid) == 0:
            plt.close(fig)
            raise ValueError("No valid birth-date values available for the date-of-birth plot.")

        _, bin_edges, _ = hist_density_weights(
            ax,
            x[valid],
            weights=w[valid],
            bins=bins_birth,
            label="All",
            alpha=0.3,
        )

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        mask_women = (g == female_value) & valid
        mask_men = (g == male_value) & valid

        counts_women = _density_by_group(x, w, mask_women, bin_edges)
        counts_men = _density_by_group(x, w, mask_men, bin_edges)

        ax.plot(bin_centers, counts_women, linewidth=1, linestyle="--", label="Women")
        ax.plot(bin_centers, counts_men, linewidth=1, linestyle="-", label="Men")

        styled_axes(ax, xbins=5, ybins=5)
        ax.set_xlabel("Date of birth")
        ax.set_ylabel("Probability density")
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.legend()

        fig.tight_layout()
        output_path = output_folder / f"{filename}.pdf"
        fig.savefig(output_path)
        saved_files.append(output_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved_files

def weighted_mean(values, weights):
    """
    Compute a weighted mean while ignoring NaN values in both values and weights.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = (~np.isnan(values)) & (~np.isnan(weights))

    if mask.sum() == 0:
        return np.nan

    total_weight = np.sum(weights[mask])

    if total_weight == 0:
        return np.nan

    return np.sum(values[mask] * weights[mask]) / total_weight


def compute_average_age_table(
    dfs_to_plot,
    years,
    df_name,
    age_column="age",
    weight_column=None,
    gender_column="Sex",
    female_value=1,
    male_value=0,
):
    """
    Compute the weighted average age for each dataframe/year,
    for the whole population and by gender.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns:
        dataframe_name, year, group, average_age
    """

    rows = []

    for df_i, year_i in zip(dfs_to_plot, years):
        if age_column not in df_i.columns:
            raise ValueError(f"Column '{age_column}' is missing from the dataframe.")

        if weight_column is None:
            weights = np.ones(len(df_i))
        else:
            if weight_column not in df_i.columns:
                raise ValueError(f"Column '{weight_column}' is missing from the dataframe.")
            weights = df_i[weight_column].values

        # ----------------------------------------------------
        # Whole population
        # ----------------------------------------------------
        avg_population = weighted_mean(df_i[age_column].values, weights)

        rows.append(
            {
                "dataframe_name": df_name,
                "year": year_i,
                "group": "population",
                "average_age": avg_population,
            }
        )

        # ----------------------------------------------------
        # By gender
        # ----------------------------------------------------
        if gender_column not in df_i.columns:
            raise ValueError(f"Column '{gender_column}' is missing from the dataframe.")

        gender_values = df_i[gender_column].values

        gender_groups = {
            "women": female_value,
            "men": male_value,
        }

        for group_name, group_value in gender_groups.items():
            mask = gender_values == group_value

            avg_group = weighted_mean(
                df_i.loc[mask, age_column].values,
                np.asarray(weights)[mask],
            )

            rows.append(
                {
                    "dataframe_name": df_name,
                    "year": year_i,
                    "group": group_name,
                    "average_age": avg_group,
                }
            )

    return pd.DataFrame(rows)

def save_average_age_table(
    dfs_to_plot,
    years,
    df_name,
    output_folder,
    age_column="age",
    weight_column=None,
    gender_column="Sex",
    female_value=1,
    male_value=0,
    filename=None,
):
    """
    Compute and save the weighted average age table as a CSV file.

    The saved table contains:
        - population average age
        - women average age
        - men average age

    Returns
    -------
    pd.DataFrame
        The table that was saved.
    pathlib.Path
        Path to the saved CSV file.
    """

    avg_age_df = compute_average_age_table(
        dfs_to_plot=dfs_to_plot,
        years=years,
        df_name=df_name,
        age_column=age_column,
        weight_column=weight_column,
        gender_column=gender_column,
        female_value=female_value,
        male_value=male_value,
    )

    if filename is None:
        filename = f"average_age_{df_name}.csv"

    csv_path = Path(output_folder) / filename
    avg_age_df.to_csv(csv_path, index=False)

    return avg_age_df, csv_path


def run_existence_graphs(
    df,
    df_name,
    df_name_for_title,
    df_type,
    output_folder,
    df_years_to_project=None,
    df_year_observed=None,
    mapping=None,
    weight_column=None,
    weight_column_name="weight",
    bins_birth=30,
    female_value=1,
    male_value=0,
    show=False,
):
    """
    Prepare the data and produce all existence graphs.

    Returns
    -------
    dict
        Dictionary containing the generated file paths by graph type.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    (
        dfs_to_plot_time_indep,
        dfs_to_plot_time_dep,
        dfs_to_plot_years_time_dep,
    ) = prepare_dfs_to_plot(
        df=df,
        df_type=df_type,
        df_years_to_project=df_years_to_project,
        df_year_observed=df_year_observed,
        mapping=mapping,
        weight_column=weight_column,
        weight_column_name=weight_column_name,
    )

    avg_age_df, avg_age_csv_path = save_average_age_table(
        dfs_to_plot=dfs_to_plot_time_dep,
        years=dfs_to_plot_years_time_dep,
        df_name=df_name,
        output_folder=output_folder,
        age_column="age",
        weight_column=weight_column_name,
        gender_column="gender",
        female_value=female_value,
        male_value=male_value,
    )

    saved_files = {}

    saved_files["age_gender"] = plot_age_gender(
        dfs_to_plot_time_dep=dfs_to_plot_time_dep,
        dfs_to_plot_years_time_dep=dfs_to_plot_years_time_dep,
        df_name=df_name,
        df_name_for_title=df_name_for_title,
        output_folder=output_folder,
        weight_column_name=weight_column_name,
        bins_birth=bins_birth,
        female_value=female_value,
        male_value=male_value,
        show=show,
    )

    saved_files["dob_gender"] = plot_dob_gender(
        dfs_to_plot_time_indep=dfs_to_plot_time_indep,
        df_name=df_name,
        df_name_for_title=df_name_for_title,
        output_folder=output_folder,
        weight_column_name=weight_column_name,
        bins_birth=bins_birth,
        female_value=female_value,
        male_value=male_value,
        show=show,
    )

    saved_files["average_age"] = avg_age_csv_path

    return saved_files
