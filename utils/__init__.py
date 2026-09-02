import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
DATA = CODE_DIR / "data"
RESULTS = CODE_DIR / "results"

if "MPLBACKEND" not in os.environ and not sys.platform.startswith("win"):
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        os.environ["MPLBACKEND"] = "Agg"

FIGURE_FORMATS = ("pdf",)

FIGURE_RC = {
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 15,
}


def apply_style():
    """Put FIGURE_RC into matplotlib's defaults, once, for whoever is about to draw."""
    import matplotlib

    matplotlib.rcParams.update(FIGURE_RC)

    return FIGURE_RC


def save_figure(figure, path, dpi=150, formats=FIGURE_FORMATS, close=True):
    """Write one figure to `path` in every format in `formats`, and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []

    for suffix in formats:
        target = path.with_suffix(f".{suffix}")
        figure.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(target)

    if close:
        from matplotlib import pyplot as plt
        plt.close(figure)

    return written


def save_plot_data(path, frame, **constants):
    """Write exactly what a figure draws to <figure>.csv"""
    import pandas as pd

    frame = pd.DataFrame(frame)
    for name, value in constants.items():
        frame[name] = value

    target = Path(path).with_suffix(".csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)

    return target
