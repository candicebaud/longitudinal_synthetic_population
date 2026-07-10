"""Baud Candice
Fri July 10 10:33:00 2026"""
# graph style matplotlib style for consistent figures (histograms + general plots)
# Usage:
#   import numpy as np
#   from style import setup_style, hist_density
#   setup_style()
#   fig, ax = plt.subplots()
#   hist_density(ax, data, bins=30, label="My variable")
#   ax.set_xlabel("x")
#   ax.set_ylabel("Probability")
#   ax.legend()
#   fig.savefig("hist.pdf") 

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def setup_style(
    *,
    column: str = "one",
    fontsize: int = 8,
    use_tex: bool = False,
):
    if column not in {"one", "two"}:
        raise ValueError("column must be 'one' or 'two'")

    fig_width_in = 3.5 if column == "one" else 7.16
    golden = (5**0.5 - 1) / 2
    fig_height_in = fig_width_in * golden

    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Figure sizing
        "figure.figsize": (fig_width_in, fig_height_in),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.02,

        # Fonts
        "font.family": "serif",
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,

        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",

        # Lines
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,

        # Axes
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        # Legend
        "legend.frameon": False,

        # No grid
        "axes.grid": False,
    })

def styled_axes(ax: plt.Axes, *, xbins: int | None = None, ybins: int | None = None):
    """
    Consistent axis formatting:
    - integer-ish number of ticks
    - minor ticks on
    """
    if xbins is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    if ybins is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))
    ax.minorticks_on()
    return ax


def hist_density(
    ax: plt.Axes,
    x: np.ndarray,
    *,
    bins: int | str | np.ndarray = 30,
    range=None,
    label: str | None = None,
    alpha: float = 0.35,
    edge: bool = True,
    linewidth: float = 0.8,
):
    """
    Histogram normalized to integrate to 1 (density=True).

    Returns (counts, bin_edges, patches).
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]

    hist_kws = dict(density=True, bins=bins, range=range, alpha=alpha, label=label)

    if edge:
        hist_kws.update(dict(edgecolor="black", linewidth=linewidth))

    return ax.hist(x, **hist_kws)


# def hist_with_density(
#     ax,
#     data,
#     *,
#     bins=30,
#     x_grid=None,
#     density_values=None,
#     hist_label="Sample",
#     density_label="True density",
#     hist_alpha=0.35,
#     line_width=1.4,
#     hist_color=None,      # NEW
#     density_color=None,   # NEW
# ):
#     """
#     Plots:
#       - Histogram normalized to area = 1
#       - Optional density curve overlay

#     Parameters
#     ----------
#     ax : matplotlib Axes
#     data : array-like
#     bins : int or sequence
#     x_grid : array for density x values
#     density_values : array for density(x)
#     """

#     data = np.asarray(data)
#     data = data[np.isfinite(data)]

#     # Histogram (normalized)
#     ax.hist(
#         data,
#         bins=bins,
#         density=True,
#         alpha=hist_alpha,
#         edgecolor="black",
#         linewidth=0.8,
#         label=hist_label,
#         color=hist_color,
#     )

#     # Overlay density if provided
#     if x_grid is not None and density_values is not None:
#         ax.plot(
#             x_grid,
#             density_values,
#             linewidth=line_width,
#             label=density_label,
#             color=density_color,
#         )

#     return ax

def hist_with_density(
    ax,
    data,
    *,
    bins=30,
    x_grid=None,
    density_values=None,
    hist_label="Sample",
    density_label="True density",
    hist_alpha=0.35,
    line_width=1.4,
):
    """
    Plots:
      - Histogram normalized to area = 1
      - Optional density curve overlay

    Parameters
    ----------
    ax : matplotlib Axes
    data : array-like
    bins : int or sequence
    x_grid : array for density x values
    density_values : array for density(x)
    """

    data = np.asarray(data)
    data = data[np.isfinite(data)]

    # Histogram (normalized)
    ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=hist_alpha,
        edgecolor="black",
        linewidth=0.8,
        label=hist_label,
    )

    # Overlay density if provided
    if x_grid is not None and density_values is not None:
        ax.plot(
            x_grid,
            density_values,
            linewidth=line_width,
            label=density_label,
        )

    return ax


def hist_with_theoretical_line(
    ax,
    empirical_data,
    theoretical_data,
    *,
    bins=30,
    bin_centers=None,
    hist_label="Sample",
    density_label="Approximate True Density",
    hist_alpha=0.35,
    line_width=1.4,
    line_color="#FFA500",  # Set the color of the theoretical line to orange (#FFA500)
    legend_fontsize=8
):
    """
    Plots:
      - Empirical data as a histogram with user-supplied bin heights
      - Theoretical data as a line overlaid on top

    Parameters
    ----------
    ax : matplotlib Axes
        The axis on which to plot.
    empirical_data : array-like
        The empirical data heights (i.e., the height of the bars for each bin).
    theoretical_data : array-like
        The theoretical density values (the heights of the line).
    bins : int or sequence
        Number of bins or bin edges.
    bin_centers : array for bin centers (optional)
        If not provided, it will be calculated as the midpoints of the bins.
    hist_label : str
        Label for the empirical histogram.
    density_label : str
        Label for the theoretical density line.
    hist_alpha : float
        Transparency for the histogram bars.
    line_width : float
        Line width for the density line.
    line_color : str
        Color for the theoretical line (default is orange).
    """
    
    # Empirical data should correspond to the heights of the bins
    empirical_data = np.asarray(empirical_data)
    empirical_data = empirical_data[np.isfinite(empirical_data)]

    # If bin_centers are not provided, calculate them from the bin edges
    if bin_centers is None:
        bin_centers = (bins[:-1] + bins[1:]) / 2 
    
    # empirical_centers = (empirical_data[:-1] + empirical_data[1:])/2
    # theoretical_centers = (theoretical_data[:-1] + theoretical_data[1:])/2

    # print(len(empirical_centers))
    # print(len(theoretical_centers))
    # print(len(bin_centers))

    # Plot the empirical data as a bar plot (not a standard histogram, as we provide the heights)
    ax.bar(
        bin_centers, 
        empirical_data, 
        width=np.diff(bins), 
        alpha=hist_alpha, 
        edgecolor="black", 
        linewidth=0.8, 
        label=hist_label, 
        align='center'
    )

    # Plot the theoretical data as a line
    ax.plot(
        bin_centers,
        theoretical_data,  # Theoretical density values for each bin center
        linewidth=line_width,
        label=density_label,
        color=line_color,  # The color for the theoretical line (orange)
    )

    ax.legend(fontsize=legend_fontsize)

    return ax


def hist_density_weights(
    ax: plt.Axes,
    x: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    bins: int | str | np.ndarray = 30,
    range=None,
    label: str | None = None,
    alpha: float = 0.35,
    edge: bool = True,
    linewidth: float = 0.8,
):
    """
    Weighted histogram normalized to integrate to 1 (density=True).

    Parameters
    ----------
    weights : array-like, optional
        Same length as x. If provided, histogram is weighted.

    Returns (counts, bin_edges, patches).
    """
    x = np.asarray(x)
    mask = np.isfinite(x)

    x = x[mask]
    if weights is not None:
        weights = np.asarray(weights)[mask]

    hist_kws = dict(
        density=True,
        bins=bins,
        range=range,
        alpha=alpha,
        label=label,
    )

    if weights is not None:
        hist_kws["weights"] = weights

    if edge:
        hist_kws.update(dict(edgecolor="black", linewidth=linewidth))

    return ax.hist(x, **hist_kws)

def bar_percentages(
    ax: plt.Axes,
    values,
    *,
    weights=None,
    order: str = "decreasing",
    label_map: dict | None = None,
    color=None,
    alpha: float = 0.35,
    edge: bool = True,
    linewidth: float = 0.8,
    show_values: bool = False,
    value_fmt: str = "{:.1f}",
    rotation: int = 45,
):
    """
    Categorical bar plot normalized to weighted percentages.
    Designed to match the style of hist_density().
    """

    s = pd.Series(values)

    if weights is None:
        w = pd.Series(np.ones(len(s)), index=s.index)
    else:
        w = pd.Series(weights, index=s.index)

    df_tmp = pd.DataFrame({
        "category": s,
        "weight": w,
    }).dropna(subset=["category", "weight"])

    if label_map is not None:
        df_tmp["category"] = df_tmp["category"].map(label_map)

    weighted_counts = df_tmp.groupby("category")["weight"].sum()

    percentages = weighted_counts / weighted_counts.sum() * 100

    if order == "decreasing":
        percentages = percentages.sort_values(ascending=False)
    elif order == "increasing":
        percentages = percentages.sort_values(ascending=True)

    bar_kws = dict(
        alpha=alpha,
        color=color,
    )

    if edge:
        bar_kws.update(dict(edgecolor="black", linewidth=linewidth))

    ax.bar(
        percentages.index.astype(str),
        percentages.values,
        width=0.9,
        **bar_kws,
    )

    ax.set_ylabel("Share (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, percentages.max() * 1.10)

    ax.tick_params(axis="x", rotation=rotation)

    if show_values:
        for i, v in enumerate(percentages.values):
            ax.text(
                i,
                v,
                value_fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=mpl.rcParams["font.size"],
            )

    styled_axes(ax, ybins=5)

    return percentages

def hist_categorical_interaction(
    ax: plt.Axes,
    x,
    *,
    weights=None,
    fill_by=None,
    normalize=True,
    rotation=90,
    tick_fontsize=6,
    alpha=0.35,
    show_values=True,
    value_fontsize=5,
    value_format="{:.1f}",
    ylim=None,
    show_legend=True,
):
    x = np.asarray(x)

    # Remove missing values in x
    mask = x != None

    if fill_by is not None:
        fill_by = np.asarray(fill_by)
        mask = mask & (fill_by != None)

    x = x[mask]

    if weights is not None:
        weights = np.asarray(weights)
        weights = weights[mask]
    else:
        weights = np.ones(len(x))

    if fill_by is not None:
        fill_by = fill_by[mask]

    # Total weight by main category
    categories = np.unique(x)

    counts = np.array([
        weights[x == cat].sum()
        for cat in categories
    ], dtype=float)

    # Sort categories by decreasing total weight
    order = np.argsort(-counts)
    categories = categories[order]
    counts = counts[order]

    # Normalize total bar heights
    if normalize:
        total = counts.sum()
        counts_plot = 100 * counts / total
        ax.set_ylabel("Percentage")
    else:
        total = counts.sum()
        counts_plot = counts
        ax.set_ylabel("Count")

    positions = np.arange(len(categories))

    color_cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    if fill_by is None:
        # Simple histogram
        ax.bar(
            positions,
            counts_plot,
            edgecolor="black",
            linewidth=0.8,
            alpha=alpha,
            color=color_cycle[0],
        )

    else:
        # Stacked histogram
        fill_categories = np.unique(fill_by)

        bottom = np.zeros(len(categories))

        for k, fill_cat in enumerate(fill_categories):
            segment_values = []

            for cat in categories:
                mask_cat = x == cat
                mask_fill = fill_by == fill_cat

                value = weights[mask_cat & mask_fill].sum()

                if normalize:
                    value = 100 * value / total

                segment_values.append(value)

            segment_values = np.array(segment_values)

            ax.bar(
                positions,
                segment_values,
                bottom=bottom,
                edgecolor="black",
                linewidth=0.8,
                alpha=alpha,
                color=color_cycle[k % len(color_cycle)],
                label=str(fill_cat),
            )

            bottom += segment_values

        if show_legend:
            ax.legend(fontsize=value_fontsize)

    # Values on top of total bars
    if show_values:
        for pos, value in zip(positions, counts_plot):
            ax.text(
                pos,
                value,
                value_format.format(value),
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
            )

    # Fixed or automatic y-axis limit
    if ylim is not None:
        ax.set_ylim(ylim)
    elif len(counts_plot) > 0:
        ax.set_ylim(0, max(counts_plot) * 1.12)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        categories,
        rotation=rotation,
        ha="right",
        fontsize=tick_fontsize,
    )

    return categories, counts_plot