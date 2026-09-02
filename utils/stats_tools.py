from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy import stats

from . import apply_style, save_figure, save_plot_data

apply_style()


def _absorb(letters):
    """Drop empty and duplicate letters, and any letter another one already covers."""
    unique = {frozenset(members) for members in letters if members}

    return [set(members) for members in unique if not any(members < other for other in unique)]

def _compact_letters(separable, means):
    """Letters from a boolean separable[i, j]. Two groups share one when it cannot separate."""
    count = len(means)

    letters = [set(range(count))]
    for i in range(count):
        for j in range(i + 1, count):
            if not separable[i][j]:
                continue

            for members in [held for held in letters if i in held and j in held]:
                letters.remove(members)
                letters += [members - {i}, members - {j}]

            letters = _absorb(letters)

    # 'a' goes to the letter holding the highest mean, so the display reads top-down
    letters.sort(key=lambda members: -max(means[i] for i in members))

    assigned = [""] * count
    for position, members in enumerate(letters):
        for i in members:
            assigned[i] += chr(ord("a") + position)

    return assigned

def block_design(frame, arm_column, value_column, block_column="seed", warn=True):
    """[arms, blocks] matrix and its arm names, keeping only fully observed blocks."""
    wide = frame.pivot(index=arm_column, columns=block_column, values=value_column)
    complete = wide.dropna(axis=1, how="any")

    dropped = wide.shape[1] - complete.shape[1]
    if dropped and warn:
        missing = [b for b in wide.columns if b not in complete.columns]
        print(f"  [block_design] {value_column}: dropped {dropped} incomplete "
                f"{block_column}(s) {missing}, {complete.shape[1]} left")

    return complete.to_numpy(), list(complete.index), list(complete.columns)

def _block_error(values):
    """Residual mean square and df for y_ij = mu + arm_i + block_j + e_ij."""
    arms, blocks = values.shape
    residual = (values - values.mean(1, keepdims=True) - values.mean(0, keepdims=True)
                + values.mean())

    df = (arms - 1) * (blocks - 1)

    if df < 1:
        return float("nan"), df

    return float((residual ** 2).sum()) / df, df

def blocked_tukey_cld(values, names=None, alpha=0.05):
    """Compact letters from a randomized complete block design: [arms, blocks]."""
    values = np.asarray(values, dtype=float)
    arms, blocks = values.shape

    if names is None:
        names = [str(i) for i in range(arms)]

    means = values.mean(1)
    mse, df = _block_error(values)

    if df < 1 or not np.isfinite(mse):
        return {name: {"mean": means[i], "letters": "", "blocks": blocks}
                for i, name in enumerate(names)}

    # a shared error term, so one critical difference covers every pair
    critical = stats.studentized_range.ppf(1 - alpha, arms, df) * np.sqrt(mse / blocks)
    separable = np.abs(means[:, None] - means[None, :]) > critical

    assigned = _compact_letters(separable, means)

    return {name: {"mean": means[i], "letters": assigned[i], "blocks": blocks,
                    "critical_difference": critical}
            for i, name in enumerate(names)}

@lru_cache(maxsize=None)
def _dunnett_null(arms, df, draws=200000, seed=0):
    """Sorted max |t| under the null, shared by the critical value and the p-values."""
    rng = np.random.default_rng(seed)

    normal = rng.standard_normal((draws, arms))
    scale = np.sqrt(rng.chisquare(df, draws) / df)
    statistic = (normal[:, 1:] - normal[:, [0]]) / (scale[:, None] * np.sqrt(2))

    return np.sort(np.abs(statistic).max(axis=1))

def _dunnett_critical(arms, df, alpha=0.05):
    """Two-sided Dunnett critical value: max |t| over arms-1 comparisons to one control."""
    return float(np.quantile(_dunnett_null(arms, df), 1 - alpha))

def _dunnett_pvalue(statistic, arms, df):
    """Family-wise p under Dunnett's null, so several metrics can be corrected together."""
    null = _dunnett_null(arms, df)
    beyond = null.size - np.searchsorted(null, abs(statistic), side="left")

    # never report an exact zero
    return float(max(beyond, 1) / null.size)

def blocked_dunnett(values, names, control, alpha=0.05):
    """Every arm against one control, on a block design"""
    values = np.asarray(values, dtype=float)
    arms, blocks = values.shape

    if control not in names:
        raise ValueError(f"control {control!r} is not one of {names}")

    index = names.index(control)
    means = values.mean(1)
    mse, df = _block_error(values)

    if df < 1 or not np.isfinite(mse) or mse <= 0:
        return pd.DataFrame()

    spread = np.sqrt(2 * mse / blocks)
    critical = _dunnett_critical(arms, df, alpha)

    rows = []
    for i, name in enumerate(names):
        if i == index:
            continue

        difference = means[i] - means[index]
        rows.append({
            "arm": name, "control": control,
            "mean": means[i], "control_mean": means[index],
            "difference": difference,
            "t": difference / spread,
            "critical": critical,
            "critical_difference": critical * spread,
            "cohens_d_within_block": difference / np.sqrt(mse),
            "p": _dunnett_pvalue(difference / spread, arms, df),
            "significant": abs(difference) > critical * spread,
            "blocks": blocks,
        })

    return pd.DataFrame(rows).sort_values("difference", ascending=False)

def adjust_family(contrasts, family="primary", alpha=0.05, column="family"):
    """Benjamini-Hochberg across the confirmatory contrasts, leaving the rest untouched."""
    if contrasts.empty or column not in contrasts:
        return contrasts

    contrasts = contrasts.copy()
    contrasts["p_adjusted"] = np.nan
    contrasts["significant_adjusted"] = False

    primary = contrasts.index[contrasts[column] == family]
    if len(primary):
        adjusted, rejected = benjamini_hochberg(contrasts.loc[primary, "p"], alpha)
        contrasts.loc[primary, "p_adjusted"] = adjusted
        contrasts.loc[primary, "significant_adjusted"] = rejected

    return contrasts

def benjamini_hochberg(pvalues, alpha=0.05):
    """Adjusted p-values and rejections at false discovery rate `alpha`."""
    pvalues = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(pvalues)
    count = int(finite.sum())
    out = np.full(pvalues.shape, np.nan, dtype=float)

    if count == 0:
        return out, np.zeros(pvalues.shape, dtype=bool)

    finite_values = pvalues[finite]
    order = np.argsort(finite_values)
    ranked = finite_values[order]

    steps = np.arange(1, count + 1)
    # step up from the largest p, so the adjusted values stay monotone
    adjusted = np.minimum.accumulate((ranked * count / steps)[::-1])[::-1]

    finite_out = np.empty(count)
    finite_out[order] = np.minimum(adjusted, 1.0)
    out[finite] = finite_out

    return out, np.isfinite(out) & (out <= alpha)

def paired_comparison(frame, group_column, baseline, treatment, metrics, pair_on="seed", by=None, alpha=0.05):
    """Paired t-test of two levels of group_column, one row per metric."""
    # paired because a seed fixes the split as well as the initialisation
    groups = [(None, frame)] if by is None else list(frame.groupby(by))
    label = by or "group"
    rows = []

    for group_name, group in groups:
        # pivot, not pivot_table: a repeated cell is a duplicated run and has to raise
        wide = group.pivot(index=pair_on, columns=group_column, values=metrics)

        for metric in metrics:
            if not {baseline, treatment} <= set(wide[metric].columns):
                continue

            pair = wide[metric][[baseline, treatment]].dropna()
            if len(pair) < 2:
                continue

            before = pair[baseline].to_numpy()
            after = pair[treatment].to_numpy()
            difference = after - before
            spread = difference.std(ddof=1)
            shift = difference.mean()

            if spread:
                statistic, pvalue = stats.ttest_rel(after, before)
                effect = shift / spread
            elif shift:
                statistic, pvalue, effect = np.inf, 0.0, np.sign(shift) * np.inf
            else:
                statistic, pvalue, effect = 0.0, 1.0, 0.0

            rows.append({
                label: group_name,
                "metric": metric,
                "n": len(pair),
                baseline: before.mean(),
                treatment: after.mean(),
                "difference": shift,
                "cohens_d": effect,
                "t": statistic,
                "p": pvalue,
            })

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results["p_adjusted"] = np.nan
    results["significant"] = False

    for index in results.groupby(label, dropna=False).groups.values():
        adjusted, rejected = benjamini_hochberg(results.loc[index, "p"], alpha)
        results.loc[index, "p_adjusted"] = adjusted
        results.loc[index, "significant"] = rejected

    return results.sort_values([label, "p"]).reset_index(drop=True)

def _finite(values, name):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]

    if values.size < 3:
        raise ValueError(f"need at least 3 finite values in {name}, got {values.size}")

    return values

def qq_plot(values, ax=None, distribution="norm", path=None, figsize=(4.5, 4.5)):
    """Q-Q plot of values against a theoretical distribution, with a fitted line."""
    values = _finite(values, "values")

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=figsize)

    (theoretical, ordered), (slope, intercept, r) = stats.probplot(values, dist=distribution, fit=True)

    ax.scatter(theoretical, ordered, s=10, alpha=0.5, edgecolor="none", rasterized=True)
    ax.plot(theoretical, slope * theoretical + intercept, color="crimson", linewidth=1)

    ax.set_xlabel(f"Theoretical quantiles ({distribution})")
    ax.set_ylabel("Ordered values")
    ax.annotate(f"$R^2$ = {r ** 2:.3f}\nn = {values.size:,}", xy=(0.05, 0.86), xycoords="axes fraction", fontsize=11)

    if path is not None:
        save_plot_data(path,
                        {"theoretical_quantile": theoretical, "ordered_value": ordered,
                        "fitted": slope * theoretical + intercept},
                        slope=slope, intercept=intercept, r_squared=r ** 2)
        save_figure(ax.figure, path, close=created)

    return ax

def sample_qq_plot(sample, reference, ax=None, sample_label="predicted",
                    reference_label="observed", path=None, figsize=(4.5, 4.5)):
    """Q-Q plot of two samples against the identity, for unequal and unpaired sizes."""
    sample = _finite(sample, "sample")
    reference = _finite(reference, "reference")

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=figsize)

    # midpoints, so neither extreme quantile lands on a single order statistic
    count = min(sample.size, reference.size)
    probabilities = (np.arange(1, count + 1) - 0.5) / count

    sample_quantiles = np.quantile(sample, probabilities)
    reference_quantiles = np.quantile(reference, probabilities)

    limits = [min(sample_quantiles[0], reference_quantiles[0]), max(sample_quantiles[-1], reference_quantiles[-1])]

    ax.plot(limits, limits, color="crimson", linewidth=1, zorder=1)
    ax.scatter(reference_quantiles, sample_quantiles, s=10, alpha=0.5, edgecolor="none",
                zorder=2, rasterized=True)

    reference_spread = reference.std(ddof=1)
    spread = sample.std(ddof=1) / reference_spread if reference_spread else np.nan
    distance = stats.ks_2samp(sample, reference)

    ax.set_xlabel(f"{reference_label[:1].upper()}{reference_label[1:]} quantiles")
    ax.set_ylabel(f"{sample_label[:1].upper()}{sample_label[1:]} quantiles")
    ax.annotate(f"spread ratio = {spread:.3f}\nKS = {distance.statistic:.3f}"
                f"\nn = {sample.size:,} vs {reference.size:,}",
                xy=(0.05, 0.80), xycoords="axes fraction", fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")

    if path is not None:
        save_plot_data(path,
                        {"probability": probabilities,
                        f"{reference_label}_quantile": reference_quantiles,
                        f"{sample_label}_quantile": sample_quantiles},
                        spread_ratio=spread, ks_statistic=distance.statistic,
                        ks_pvalue=distance.pvalue)
        save_figure(ax.figure, path, close=created)

    return ax
