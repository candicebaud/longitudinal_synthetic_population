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
        "savefig.bbox": "tight",
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