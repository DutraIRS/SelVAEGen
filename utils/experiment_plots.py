from pathlib import Path

import numpy as np
import pandas as pd

from . import apply_style, save_figure, save_plot_data
from .experiment_results import (BRANCH_AGGREGATION, DECISION_METRIC, SUCCESS_THRESHOLD,
                            branch_decision, decision_column)

apply_style()


def _axis_label(text):
    """An axis label: first letter upper, the rest left exactly as written."""
    text = str(text)

    return text[:1].upper() + text[1:]


def _save(figure, path, dpi=150):
    figure.tight_layout()
    save_figure(figure, path, dpi=dpi)

def plot_losses(frame, path):
    """Train against validation loss, with the epoch the checkpoint came from."""
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 4))

    axes.plot(frame["epoch"], frame["loss"], label="train")
    axes.plot(frame["epoch"], frame["val_loss"], label="validation")

    chosen = int(frame.get("select_score", frame["val_loss"]).idxmin())
    axes.axvline(frame["epoch"][chosen], color="grey", linestyle=":", linewidth=1,
                    label=f"checkpoint @ epoch {frame['epoch'][chosen]}")

    lowest = int(frame["val_loss"].idxmin())
    if lowest != chosen:
        axes.axvline(frame["epoch"][lowest], color="tab:orange", linestyle=":",
                        linewidth=1, label=f"val_loss min @ epoch {frame['epoch'][lowest]}")

    save_plot_data(path, frame[["epoch", "loss", "val_loss"]],
                    checkpoint_epoch=int(frame["epoch"][chosen]),
                    val_loss_min_epoch=int(frame["epoch"][lowest]))

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_yscale("log")
    axes.legend()

    _save(figure, path)

def plot_grad_norm(frame, path):
    """Pre-clip gradient norm against the clip threshold it is being cut to."""
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 4))

    axes.plot(frame["epoch"], frame["grad_norm"], label="mean")
    axes.plot(frame["epoch"], frame["grad_norm_max"], label="max", alpha=0.5)

    if "grad_clip" in frame:
        clip = frame["grad_clip"].iloc[0]
        share = frame["grad_clipped"].mean()
        axes.axhline(clip, color="crimson", linestyle="--", linewidth=1, label=f"clip = {clip:g} ({share:.0%} of steps)")

    columns = [c for c in ("epoch", "grad_norm", "grad_norm_max", "grad_clipped", "grad_clip", "batches_skipped") if c in frame]
    save_plot_data(path, frame[columns])

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Gradient norm (pre-clip)")
    axes.set_yscale("log")
    axes.legend()

    _save(figure, path)

HEADLINE_METRICS = {
    "selectivity": ("delta_score", "distributional_delta_score", "directional_consistency",
                    "tanimoto_gap", "effective_delta"),
    "quality": ("validity", "uniqueness", "novelty"),
    "potency_chemistry": ("average_pba", "average_top10", "success_rate", "ro5", "qed",
                            "synthetic_accessibility", "internal_diversity", "fcd"),
}

BOUNDED_METRICS = ("validity", "strict_validity", "uniqueness", "novelty", "yield",
                    "directional_consistency", "internal_diversity", "success_rate",
                    "qed", "ro5")

NULL_LINES = {"directional_consistency": 0.5, "delta_score": 0.0, "tanimoto_gap": 0.0,
                "distributional_delta_score": 0.0, "effective_delta": 0.0}


THRESHOLD_METRICS = ("average_pba", "average_top10")

def _metric_bounds(column, threshold=None):
    """(ylim, reference line) for a column, each None where it does not apply."""
    limits = (-0.02, 1.02) if column.endswith(BOUNDED_METRICS) else None
    if column.endswith("directional_consistency"):
        limits = (0.25, 0.75)

    if threshold is not None and column.endswith(THRESHOLD_METRICS):
        return limits, float(threshold)

    null = next((value for name, value in
                    sorted(NULL_LINES.items(), key=lambda pair: -len(pair[0]))
                    if column.endswith(name)), None)

    return limits, null

def frame_threshold(frame, column="dataset"):
    """The potency threshold for the frame's dataset, or None if it holds more than one."""
    if column not in frame:
        return None

    names = frame[column].dropna().unique()

    return SUCCESS_THRESHOLD.get(names[0]) if len(names) == 1 else None

METRIC_LABELS = {
    "delta_score": r"$\Delta$ score",
    "distributional_delta_score": r"Distributional $\Delta$ score",
    "directional_consistency": "Directional consistency",
    "tanimoto_gap": "Tanimoto gap",
    "effective_delta": r"Effective $\Delta$",
    "selectivity_effect": r"Selectivity effect $d$",
    "success_rate": "Success rate",
    "average_pba": "Mean predicted affinity",
    "average_top10": "Mean of top decile",
    "hit_delta_score": r"$\Delta$ score among hits",
    "validity": "Validity",
    "strict_validity": "Strict validity",
    "uniqueness": "Uniqueness",
    "novelty": "Novelty",
    "yield": "Yield",
    "qed": "QED",
    "ro5": "Lipinski Ro5",
    "synthetic_accessibility": "SA score",
    "internal_diversity": "Internal diversity",
    "mean_atoms_ratio": "Size / training size",
    "fcd": "FCD",
}

def _metric_label(name):
    """Panel title for a metric, capitalised, falling back to the name itself."""
    label = METRIC_LABELS.get(name) or name.replace("_", " ")

    return label[:1].upper() + label[1:]

BOX_PALETTE = ["#6b8cc7", "#7fb08a", "#d3a04f", "#b57ea8", "#5fa8a0",
                "#c78b6b", "#8f8fbf", "#a8b06b", "#c76b7e"]

def _panel_grid(count, width=3.6, height=3.7, max_row=5, columns=3):
    """(figure, flat axes) for `count` panels: one row up to `max_row`, else `columns` wide."""
    from matplotlib import pyplot as plt

    if isinstance(columns, tuple):
        tall, wide = columns
    else:
        wide = count if count <= max_row else columns
        tall = -(-count // wide)                  # ceiling division

    figure, axes = plt.subplots(tall, wide, figsize=(width * wide, height * tall))
    flat = list(np.atleast_1d(axes).ravel())

    for spare in flat[count:]:
        spare.set_visible(False)

    return figure, flat[:count]

def _metric_group_grid(group, count):
    """The shared layout for metric boxes and metric curves."""
    if group == "potency_chemistry" and count == 8:
        return _panel_grid(count, columns=(2, 4))

    return _panel_grid(count)

def _metric_panel(axes, frame, arm_column, column, order, label=None, threshold=None):
    """One metric: a box per arm, with every seed drawn on its own box."""
    values, names = [], []
    for arm in order:
        seeds = frame.loc[frame[arm_column] == arm, column].to_numpy(dtype=float)
        seeds = seeds[np.isfinite(seeds)]
        if seeds.size:
            values.append(seeds)
            names.append(arm)

    if not values:
        axes.set_visible(False)
        return False

    positions = np.arange(len(names))
    boxes = axes.boxplot(values, positions=positions, widths=0.62, patch_artist=True,
                            medianprops={"color": "#11181f", "linewidth": 1.6},
                            whiskerprops={"color": "#67727e"},
                            capprops={"color": "#67727e"}, showfliers=False)

    for patch, colour in zip(boxes["boxes"], BOX_PALETTE * 4):
        patch.set(facecolor=colour, alpha=0.42, edgecolor=colour, linewidth=1.3)

    jitter = np.random.default_rng(0)
    for position, seeds in zip(positions, values):
        axes.scatter(position + jitter.uniform(-0.14, 0.14, seeds.size), seeds, s=13,
                        zorder=3, color="#2c3540", alpha=0.72, linewidths=0)

    limits, null = _metric_bounds(column, threshold)
    if limits is not None:
        axes.set_ylim(*limits)
    if null is not None:
        axes.axhline(null, color="#b0392b", linestyle="--", linewidth=0.9, alpha=0.7,
                        zorder=1)

    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
    axes.set_ylabel(_axis_label(label or column), fontsize=13)
    axes.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)

    return True

def describe_arms(frame, arm_column, path=None):
    """mean, std and n of every recorded metric per arm, with no test or correction."""
    columns = [c for c in frame.columns if c.startswith(("seq_", "graph_"))
                and not c.endswith(("_reference_mean_atoms", "_n_targets", "_n"))]
    if not columns:
        return pd.DataFrame()

    grouped = frame.groupby(arm_column)[columns]
    summary = (grouped.agg(["mean", "std", "count"])
                .stack(level=0, future_stack=True)
                .reset_index()
                .rename(columns={"level_1": "metric", "count": "n"}))
    summary["reported"] = [f"{m:.4f} +- {s:.4f}" if pd.notna(s) else f"{m:.4f}"
                            for m, s in zip(summary["mean"], summary["std"])]

    if path is not None:
        summary.to_csv(path, index=False)

    return summary

# names that describe combining the branches, as against restricting to one of them
AGGREGATIONS = ("mean", "max", "min", "median", "sum")


def plot_decision(frame, arm_column, path, order=None, metric=None, aggregation=None,
                    branches=None):
    """The number the ablation is decided on: a box per arm, with every seed drawn on it."""
    from matplotlib import pyplot as plt

    names = ([metric or DECISION_METRIC] if metric is None or isinstance(metric, str)
                else list(metric))
    aggregation = aggregation or BRANCH_AGGREGATION
    resolve = {} if branches is None else {"branches": tuple(branches)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = {name: (frame[name] if name in frame
                        else branch_decision(frame, name, **resolve)) for name in names}
    table = pd.DataFrame({arm_column: frame[arm_column], "seed": frame.get("seed"),
                            **columns}).dropna(subset=names, how="all")
    arms = list(order) if order else sorted(table[arm_column].unique())
    arms = [a for a in arms if (table[arm_column] == a).any()]
    if len(arms) < 2:
        return None

    note = f"  ({aggregation} over branches)" if aggregation in AGGREGATIONS else ""
    figure, cells = plt.subplots(1, len(names), squeeze=False, figsize=((1.5 * len(arms) + 3) * len(names), 5))

    drawn = []
    for cell, name in zip(cells[0], names):
        if _metric_panel(cell, table.dropna(subset=[name]), arm_column, name, arms,
                            f"{name}{note}"):
            cell.set_ylabel(f"{_metric_label(name)}{note}", fontsize=13)
            drawn.append(name)
        else:
            cell.set_visible(False)

    if not drawn:
        plt.close(figure)
        return None

    figure.tight_layout()
    save_plot_data(path, table, metric=", ".join(names), aggregation=aggregation)
    _save(figure, path, dpi=200)

    return path

def plot_metric_boxes(frame, arm_column, directory, branches=("seq", "graph"),
                        groups=None, order=None, oracle="MeanOracle"):
    """One figure per decoder branch per metric group: a box per arm, every seed on it."""
    from matplotlib import pyplot as plt

    groups = HEADLINE_METRICS if groups is None else groups
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    threshold = frame_threshold(frame)
    written = []

    for branch in branches:
        for group, metrics in groups.items():
            panels = [(decision_column(frame, branch, name, oracle),
                        _metric_label(name))
                        for name in metrics]
            panels = [(column, label) for column, label in panels if column is not None]
            if not panels:
                continue

            columns = [column for column, _ in panels]

            arms = list(order) if order else sorted(frame[arm_column].dropna().unique())
            # an arm without this decoder has nothing to show in this branch
            arms = [arm for arm in arms
                    if frame.loc[frame[arm_column] == arm, columns].notna().any().any()]
            if len(arms) < 2:
                continue

            figure, cells = _metric_group_grid(group, len(panels))
            drawn = [_metric_panel(one, frame, arm_column, column, arms, label, threshold)
                        for one, (column, label) in zip(cells, panels)]

            if not any(drawn):
                plt.close(figure)
                continue

            path = directory / f"{branch}_{group}.pdf"
            save_plot_data(path, frame[[arm_column, "seed"] + columns])
            _save(figure, path)
            written.append(path)

    return written

LINE_PALETTE = ["#2f6fb5", "#3f8f5c", "#c08a2e", "#9a5ba0", "#3f938c",
                "#b06a45", "#6b6bb0", "#87923f", "#b0455c"]

def _curve_panel(axes, frame, x_column, arm_column, column, arms, label=None,
                    confidence=0.95, reference=None, threshold=None):
    """One metric against `x_column`: a mean line per arm, banded over the blocks."""
    from scipy import stats

    drawn = False
    for index, arm in enumerate(arms):
        block = frame[frame[arm_column] == arm]
        grouped = block.groupby(x_column)[column].agg(["mean", "std", "count"]).dropna(
            subset=["mean"])
        if grouped.empty:
            continue

        x = grouped.index.to_numpy(dtype=float)
        mean = grouped["mean"].to_numpy(dtype=float)
        colour = LINE_PALETTE[index % len(LINE_PALETTE)]

        # a band from three seeds is wide and should look it: Student's t, not 1.96
        n = grouped["count"].to_numpy(dtype=float)
        half = np.where(n > 1,
                        stats.t.ppf(0.5 + confidence / 2, np.maximum(n - 1, 1))
                        * grouped["std"].to_numpy(dtype=float) / np.sqrt(np.maximum(n, 1)),
                        np.nan)

        good = np.isfinite(half)
        if good.any():
            axes.fill_between(x[good], (mean - half)[good], (mean + half)[good],
                                color=colour, alpha=0.16, linewidth=0)

        axes.plot(x, mean, color=colour, linewidth=1.7, label=arm, zorder=3)
        # the seeds themselves, so a band drawn from three points cannot hide them
        axes.scatter(block[x_column], block[column], s=7, color=colour, alpha=0.45,
                        linewidths=0, zorder=2)
        drawn = True

    if not drawn:
        return False

    limits, null = _metric_bounds(column, threshold)
    if limits is not None:
        axes.set_ylim(*limits)
    if null is not None:
        axes.axhline(null, color="#b0392b", linestyle="--", linewidth=0.9, alpha=0.7,
                        zorder=1)
    if reference is not None:
        # where every other experiment generated, so the curve is read against it
        axes.axvline(reference, color="#67727e", linestyle=":", linewidth=1.0, alpha=0.8,
                        zorder=1)

    axes.set_ylabel(_axis_label(label or column), fontsize=13)
    axes.set_xlabel(_axis_label(x_column.replace("_", " ")), fontsize=13)
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)

    return True

def plot_metric_curves(frame, x_column, arm_column, directory, branches=("seq", "graph"),
                        groups=None, order=None, oracle="MeanOracle",
                        confidence=0.95, reference=None):
    """One figure per branch per metric group: each metric against `x_column`."""
    from matplotlib import pyplot as plt

    groups = HEADLINE_METRICS if groups is None else groups
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    threshold = frame_threshold(frame)
    written = []

    for branch in branches:
        for group, metrics in groups.items():
            panels = [(decision_column(frame, branch, name, oracle),
                        _metric_label(name))
                        for name in metrics]
            panels = [(column, label) for column, label in panels if column is not None]
            if not panels:
                continue

            columns = [column for column, _ in panels]
            arms = list(order) if order else sorted(frame[arm_column].dropna().unique())
            arms = [arm for arm in arms
                    if frame.loc[frame[arm_column] == arm, columns].notna().any().any()]
            if not arms:
                continue

            figure, cells = _metric_group_grid(group, len(panels))
            drawn = [_curve_panel(one, frame, x_column, arm_column, column, arms, label,
                                    confidence, reference, threshold)
                        for one, (column, label) in zip(cells, panels)]

            if not any(drawn):
                plt.close(figure)
                continue

            cells[0].legend(fontsize=11, frameon=False)

            path = directory / f"{branch}_{group}.pdf"
            keep = [c for c in (arm_column, x_column, "seed") if c in frame]
            save_plot_data(path, frame[keep + columns])
            _save(figure, path)
            written.append(path)

    return written
