"""
Statistical plotting utilities.

This module provides functions for creating publication-quality
statistical plots including Bland-Altman plots.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import blended_transform_factory

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "mathtext.fontset": "stix",
})


def bland_altman(
        m0: np.ndarray,
        m1: np.ndarray,
        *,
        ax: Optional[plt.Axes] = None,
        title: str = "Bland–Altman Plot",
        x_label: str = "Mean of methods",
        y_label: str = "Difference",
        units: Optional[str] = None,
        point_alpha: float = 0.7,
        show_proportional_bias: bool = True,
        loa_factor: float = 1.96,
        annotate: bool = True,
        annotate_position: Union[str, float] = "auto",
) -> Tuple[plt.Figure, plt.Axes, Dict[str, float]]:
    """
    Create a Bland–Altman plot comparing two sets of measurements.

    Supports repeated measures: if multiple trials per subject, uses
    the method of Bland & Altman (2007) to partition variance into
    between- and within-subject components for appropriate LoA.

    Parameters
    ----------
    m0 : np.ndarray
        First set of measurements. Shape (N,) for single measures or
        (N, T) for N subjects with T trials each.
    m1 : np.ndarray
        Second set of measurements. Same shape as m0.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates a new figure/axes.
    title, x_label, y_label : str
        Plot labels. If `units` is provided, axes labels get " [units]" appended.
    units : str, optional
        Physical units (e.g., "L"). Appended to axis labels if provided.
    point_alpha : float
        Alpha for scatter points.
    show_proportional_bias : bool
        If True, fit diff ~ avg and draw regression line.
    loa_factor : float
        Multiplier for limits of agreement (default 1.96 ≈ 95% LoA).
    annotate : bool
        If True, annotate mean diff and LoA on the plot.
    annotate_position : {"auto", "top", "bottom", "inside"} or float
        Vertical placement of the annotation block. "auto" (default) picks
        the side opposite the bias — bottom if mean_diff >= 0, top otherwise —
        so the text lands in the empty half of the plot. "inside" anchors
        the text just below the upper LoA line (useful when one of the LoA
        bounds is close to the axes edge). A float in [0, 1] is interpreted
        as an axes y-fraction for explicit placement.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    stats : dict
        {
          "n_subjects": int,
          "n_trials": int,
          "n_total": int,
          "mean_diff": float,
          "sd_between": float,
          "sd_within": float,
          "sd_total": float,
          "loa_lower": float,
          "loa_upper": float,
          "slope": float (if show_proportional_bias),
          "intercept": float (if show_proportional_bias)
        }

    References
    ----------
    Bland JM, Altman DG. Agreement between methods of measurement with
    multiple observations per individual. J Biopharm Stat. 2007;17(4):571-82.
    """
    m0 = np.asarray(m0)
    m1 = np.asarray(m1)

    # Handle 1D input: reshape to (N, 1)
    if m0.ndim == 1:
        m0 = m0.reshape(-1, 1)
    if m1.ndim == 1:
        m1 = m1.reshape(-1, 1)

    if m0.shape != m1.shape:
        raise ValueError(f"Shape mismatch: m0 {m0.shape} vs m1 {m1.shape}")

    n_subjects, n_trials = m0.shape

    # Compute differences
    diff = m0 - m1  # (N, T)
    avg = 0.5 * (m0 + m1)  # (N, T)

    # Remove subjects with any NaN/inf
    valid_mask = np.all(np.isfinite(diff), axis=1)
    diff = diff[valid_mask]
    avg = avg[valid_mask]
    n_subjects = diff.shape[0]

    if n_subjects == 0:
        raise ValueError("No valid subjects after filtering.")

    # Subject means
    subj_means = diff.mean(axis=1)  # (N,)

    # Bias = grand mean
    mean_diff = float(subj_means.mean())

    # Variance decomposition (Bland & Altman 2007)
    var_between = float(subj_means.var(ddof=1)) if n_subjects > 1 else 0.0

    if n_trials > 1:
        # Within-subject variance: mean of each subject's variance
        var_within = float(diff.var(axis=1, ddof=1).mean())
    else:
        var_within = 0.0

    # Total SD for single-measurement comparison
    sd_total = np.sqrt(var_between + var_within)
    loa_lower = mean_diff - loa_factor * sd_total
    loa_upper = mean_diff + loa_factor * sd_total

    # Flatten for plotting
    avg_flat = avg.ravel()
    diff_flat = diff.ravel()

    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    else:
        fig = ax.figure

    # Scatter points
    ax.scatter(avg_flat, diff_flat, alpha=point_alpha)

    # Horizontal reference lines
    ax.axhline(mean_diff, linestyle="--")
    ax.axhline(loa_lower, linestyle=":")
    ax.axhline(loa_upper, linestyle=":")

    # Light band for LoA
    y_low, y_high = sorted([loa_lower, loa_upper])
    ax.fill_between(
        [np.min(avg_flat), np.max(avg_flat)],
        y_low, y_high,
        alpha=0.08
    )

    # Optional proportional-bias line (diff ~ avg on all points)
    slope = intercept = None
    if show_proportional_bias and avg_flat.size >= 2:
        slope, intercept = np.polyfit(avg_flat, diff_flat, deg=1)
        xfit = np.linspace(np.min(avg_flat), np.max(avg_flat), 100)
        yfit = slope * xfit + intercept
        ax.plot(xfit, yfit, linewidth=1.5)

    # Labels / title
    x_lab = x_label + (f" [{units}]" if units else "")
    y_lab = y_label + (f" [{units}]" if units else "")
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_title(title)

    # Annotation
    if annotate:
        lines = [
            f"Mean diff = {mean_diff:.3g}",
            f"LoA = {loa_lower:.3g} to {loa_upper:.3g}"
        ]
        if slope is not None:
            lines.append(f"Slope = {slope:.3g}")

        position = annotate_position
        if position == "auto":
            # Place opposite the bias: positive bias clusters data upward,
            # so the bottom is empty (and vice versa).
            position = "bottom" if mean_diff >= 0 else "top"

        # Default transform: y in axes fraction.
        trans = ax.transAxes
        if isinstance(position, (int, float)) and not isinstance(position, bool):
            if not 0.0 <= float(position) <= 1.0:
                raise ValueError(
                    f"annotate_position float must be in [0, 1]; got {position!r}"
                )
            text_y = float(position)
            va = "top" if text_y > 0.5 else "bottom"
        elif position == "bottom":
            text_y, va = 0.02, "bottom"
        elif position == "top":
            text_y, va = 0.98, "top"
        elif position == "inside":
            # Anchor text top to the upper LoA line, in data coords.
            text_y, va = loa_upper, "top"
            trans = blended_transform_factory(ax.transAxes, ax.transData)
        else:
            raise ValueError(
                "annotate_position must be 'auto', 'top', 'bottom', 'inside', "
                f"or a float in [0, 1]; got {annotate_position!r}"
            )

        ax.text(
            0.02, text_y, "\n".join(lines),
            transform=trans,
            va=va, ha="left"
        )

    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()

    stats = {
        "n_subjects": n_subjects,
        "n_trials": n_trials,
        "n_total": n_subjects * n_trials,
        "mean_diff": mean_diff,
        "sd_between": np.sqrt(var_between),
        "sd_within": np.sqrt(var_within),
        "sd_total": sd_total,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
    }
    if slope is not None:
        stats.update({"slope": float(slope), "intercept": float(intercept)})

    return fig, ax, stats