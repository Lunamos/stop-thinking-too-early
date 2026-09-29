"""Shared matplotlib style for loopdyn paper figures (palette from the dataviz reference)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8984", "grid": "#e4e3df",
    "band": "#f0efec", "band2": "#e6ecf7",
}
CAT = [C[k] for k in ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]]
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.labelcolor": C["ink"],
        "axes.edgecolor": C["ink2"],
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.5,
        "xtick.color": C["ink2"],
        "ytick.color": C["ink2"],
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left", color=C["ink"])


def shade(ax, a, b, color=None, label=None, alpha=1.0):
    ax.axvspan(a, b, color=color or C["band"], zorder=0, lw=0, alpha=alpha, label=label)
