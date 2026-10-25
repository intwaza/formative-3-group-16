"""Charts for Part 1. Each function draws one chart and returns the figure."""
import matplotlib.pyplot as plt
import numpy as np

CHILD_COLOR = "#2a78d6"    # blue
FATHER_COLOR = "#eb6834"   # orange
FEMALE_COLOR = "#1baf7a"   # green
MALE_COLOR = "#4a3aa7"     # purple
BAR_COLOR = "#dcdbd4"      # light gray

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e1e0d9",
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110,
})


def draw_histogram(ax, heights):
    """Plot the height distribution as a density histogram."""
    bins = np.arange(np.floor(heights.min()), np.ceil(heights.max()) + 1)
    ax.hist(heights, bins=bins, density=True, color=BAR_COLOR, edgecolor="white", label="Data")
    ax.set_xlabel("Height (inches)")


def plot_heights(heights):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    draw_histogram(ax, heights)
    ax.set_title(f"What EM sees: {len(heights):,} heights, no labels")
    ax.set_ylabel("Density")
    return fig


def plot_fit(ax, heights, grid, child_curve, father_curve, title, lines=None):
    """Data + the two hills + their sum. `lines` = {label: x position} for vertical lines."""
    draw_histogram(ax, heights)
    ax.plot(grid, child_curve, color=CHILD_COLOR, label="Child hill")
    ax.plot(grid, father_curve, color=FATHER_COLOR, label="Father hill")
    ax.plot(grid, child_curve + father_curve, color="black", linewidth=1.5, label="Both added together")
    for (label, x), style in zip((lines or {}).items(), ["--", ":"]):
        ax.axvline(x, color="black", linestyle=style, linewidth=1.5, label=label)
    ax.set_title(title, fontsize=10)


def plot_log_likelihood(scores):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(range(len(scores)), scores, color=CHILD_COLOR, marker="o", markersize=3)
    ax.set_title("Log-likelihood after each iteration (higher = better fit)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-likelihood")
    return fig


def plot_child_probability(grid, p_child, boundary):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(grid, p_child, color=CHILD_COLOR, label="P(Child | height)")
    ax.plot(grid, 1 - p_child, color=FATHER_COLOR, label="P(Father | height)")
    ax.axvline(boundary, color="black", linestyle=":", linewidth=1.5)
    ax.annotate(f"50/50 at {boundary:.2f} in", (boundary, 0.5), xytext=(6, 6), textcoords="offset points")
    ax.set_title("Probability of each group, for any height")
    ax.set_xlabel("Height (inches)")
    ax.set_ylabel("Probability")
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.75))
    return fig


def plot_true_groups(heights, is_child, is_male):
    """The hidden labels, grouped two ways: by role and by sex."""
    bins = np.arange(np.floor(heights.min()), np.ceil(heights.max()) + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, title, groups in [
        (left, "Grouped by role", [("Children", is_child, CHILD_COLOR), ("Fathers", ~is_child, FATHER_COLOR)]),
        (right, "Grouped by sex", [("Female", ~is_male, FEMALE_COLOR), ("Male", is_male, MALE_COLOR)]),
    ]:
        for name, mask, color in groups:
            ax.hist(heights[mask], bins=bins, density=True, histtype="step", linewidth=2,
                    color=color, label=f"{name} ({mask.sum()})")
        ax.set_title(f"{title} (each group scaled to the same size)", fontsize=10)
        ax.set_xlabel("Height (inches)")
        ax.legend(loc="upper right")
    left.set_ylabel("Density")
    return fig
