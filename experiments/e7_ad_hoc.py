import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import sem, spearmanr, t
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import silhouette_score

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "experiments"))

import e6_compare_models as source_module
from utils import DATA, RESULTS, save_figure, save_plot_data
from utils.experiment_tools import experiment_dir, setup_compute

DATASETS = ["kiba", "bindingdb", "papyrus", "davis"]
SOURCE = "e6_compare"
MODEL = "SelVAEGen"            # matched against the run directory name
ORACLE = "MeanOracle"          # the other three are in the same npz
SEED = 0
YLIM = {"kiba": (9, 16)}
ORACLE_SCALE_REFERENCE = False

GENERATED_COLOUR = "#1d7874"
TRAINING_COLOUR = "#b5485d"
PAIR_OFFSET = 0.20        # how far the box for each side sits from its tick
BOX_WIDTH = 0.34
BOX_ALPHA = 0.18

STRIPE_COLOUR = "#d9d9d9"
STRIPE_ALPHA = 0.35
N_AFFINITY_PROTEINS = 20

TOP = 10              # globally most selective molecules shown
COUNTED_ONLY = True   # drop duplicates and molecules already in the training set
N_BOOT = 2000

PER_TARGET = 40       # distinct molecules per target, both panels
MAX_TARGETS = 10      # 10 x 40 = 400 rows, which still resolves on a page
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

ORACLE_MEMBERS = ("GraphDTAOracle", "GSDTAOracle", "BaselineOracle")
ORACLE_SEEDS = tuple(source_module.SEEDS)   # every seed e6 ran is computed
WRITEUP_SEED = 0                # but only this one is drawn in the thesis
ORACLE_REFERENCE_MOLECULES = 400
ORACLE_CACHE_VERSION = 1
ORACLE_BOOTSTRAPS = 2000
ORACLE_BINS = 10
BIN_BOOTSTRAPS = 500
SPLIT_SEED_E1 = 42              # e1's create_dataloaders default

AFFINITY_UNITS = {"kiba": "KIBA score"}
DEFAULT_AFFINITY_UNITS = "$pK_d$"


def affinity_units(dataset):
    return AFFINITY_UNITS.get(dataset, DEFAULT_AFFINITY_UNITS)


def usable(values_by_protein, positions, need_spread=False):
    """(position, values) for the proteins with enough molecules to draw."""
    kept = []
    for position, values in zip(positions, values_by_protein):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) > 1 and (values.std() > 0 or not need_spread):
            kept.append((position, values))

    return kept


def paired_box(values_by_protein, positions, offset, colour, width=None, alpha=0.45):
    """One protein's distribution as a box, offset from its tick."""
    kept = usable(values_by_protein, positions)
    if not kept:
        return

    parts = plt.boxplot([values for _, values in kept],
                        positions=[position + offset for position, _ in kept],
                        widths=width or BOX_WIDTH, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="#11181f", linewidth=1.0, alpha=min(1.0, alpha * 2)),
                        whiskerprops=dict(color=colour, linewidth=0.9, alpha=alpha * 1.6),
                        capprops=dict(color=colour, linewidth=0.9, alpha=alpha * 1.6))

    for patch in parts["boxes"]:
        patch.set(facecolor=colour, alpha=alpha, edgecolor=colour, linewidth=0.9)


def draw_sides(reference, generated, positions):
    """Both distributions as faded boxes, one pair per protein."""
    paired_box(reference, positions, -PAIR_OFFSET, TRAINING_COLOUR, BOX_WIDTH, BOX_ALPHA)
    paired_box(generated, positions, +PAIR_OFFSET, GENERATED_COLOUR, BOX_WIDTH, BOX_ALPHA)


def find_runs(dataset, experiment=SOURCE, model=MODEL):
    """Every run directory for one (dataset, model) that has a scored library."""
    runs = RESULTS / dataset / experiment / "runs"
    if not runs.is_dir():
        return []

    return sorted(d for d in runs.iterdir() if d.is_dir() and model in d.name and (d / "affinities.npz").exists())


def run_seed(run):
    """The split seed a run directory was trained on, from its "seed3_Model" name."""
    head = run.name.split("_", 1)[0]

    return int(head.replace("seed", "")) if head.startswith("seed") else SEED


def figure_dir(dataset, run):
    """results/<dataset>/e7_ad_hoc/<run name>/, created."""
    path = experiment_dir(dataset, "e7_ad_hoc") / run.name
    path.mkdir(parents=True, exist_ok=True)

    return path


def available_branches(run, oracle=ORACLE):
    keys = np.load(run / "affinities.npz").files
    return sorted({k.split("|")[0] for k in keys if k.count("|") == 2 and k.split("|")[1] == oracle})


def load_predictions(run, branch, oracle=ORACLE):
    """One row per (generated molecule, panel target) with its predicted affinity."""
    # scored_index row i is affinities.npz matrix row i, for the same (branch, target)
    store = np.load(run / "affinities.npz")
    panel = store["panel_ids"].astype(int)
    index = pd.read_csv(run / "scored_index.csv")
    index = index[index["branch"] == branch]

    frames = []
    for position in sorted(index["target"].unique()):
        key = f"{branch}|{oracle}|{position}"
        if key not in store.files:
            continue

        matrix = store[key]
        members = index[index["target"] == position].sort_values("row")
        if len(members) != matrix.shape[0]:
            raise RuntimeError(f"{run.name} {key}: scored_index and matrix disagree")

        width = matrix.shape[1]
        frames.append(pd.DataFrame({
            "designed_for": int(panel[position]),
            "scored_on": np.tile(panel, len(members)),
            "molecule": np.repeat([f"t{panel[position]}:{r:03d}" for r in members["row"]], width),
            "smiles": np.repeat(members["smiles"].to_numpy(), width),
            "counted": np.repeat(members["counted"].to_numpy(), width),
            "affinity": matrix.reshape(-1).astype(float),
        }))

    if not frames:
        return pd.DataFrame(), panel

    long = pd.concat(frames, ignore_index=True)
    long["on_target"] = long["scored_on"] == long["designed_for"]

    return long, panel


def ci95(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    margin = t.ppf(0.975, len(values) - 1) * sem(values)
    return values.mean() - margin, values.mean() + margin


def oracle_scale_reference(dataset, run, panel, seed=SEED, count=2000):
    """Predicted affinity of the TRAINING molecules, so both lines share one instrument."""
    import torch

    cache = run / "reference_affinities.npz"
    if cache.exists():
        store = np.load(cache)
        return pd.DataFrame(store["affinity"], columns=store["panel_ids"].astype(int))

    from utils.data_tools import create_dataloaders, load_fingerprint_pca, load_tokenizer
    from utils.experiment_tools import load_oracle, panel_loader, reference_molecules, score_with

    tokenizer = load_tokenizer(dataset, use_selfies=True)
    (train_loader, _, test_loader), _ = create_dataloaders(
        dataset, tokenizer=tokenizer, batch_size=256, train_frac=0.8, val_frac=0.1,
        test_frac=0.1, random_state=seed, groups_per_batch=16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    oracle = load_oracle(dataset, device=device)          # the same MeanOracle ensemble
    generated, _ = panel_loader(reference_molecules(train_loader, count), list(panel),
                                test_loader, tokenizer,
                                fingerprint_pca=load_fingerprint_pca(dataset), device=device)
    scores = score_with(generated, list(panel), oracle, test_loader, device).numpy()

    np.savez_compressed(cache, affinity=scores, panel_ids=np.asarray(panel))
    return pd.DataFrame(scores, columns=list(panel))


def protein_stripes(axes, positions, spacing=1.0):
    """A faint band behind every other protein."""
    for index, position in enumerate(positions):
        if index % 2:
            continue
        axes.axvspan(position - 0.5 * spacing, position + 0.5 * spacing, color=STRIPE_COLOUR, alpha=STRIPE_ALPHA, linewidth=0, zorder=0)


def affinity_per_protein(dataset, run, branch, output):
    """Generated vs training affinity, one pair of boxes per protein."""
    long, panel = load_predictions(run, branch)
    if long.empty:
        return None

    own = long[long["on_target"]]
    stats = own.groupby("scored_on")["affinity"].agg(values=list, mean="mean", n="size")
    stats[["ci_low", "ci_high"]] = pd.DataFrame([ci95(v) for v in stats["values"]], index=stats.index)
    stats = stats.sort_values("mean").reset_index()
    positions = np.linspace(0, len(stats) - 1, min(N_AFFINITY_PROTEINS, len(stats)), dtype=int)
    stats = stats.iloc[positions].reset_index(drop=True)

    if ORACLE_SCALE_REFERENCE:
        predicted = oracle_scale_reference(dataset, run, panel, seed=run_seed(run))
        reference = [predicted[p].to_numpy() for p in stats["scored_on"]]
        label = "training molecules, same oracle"
    else:
        measured = pd.read_csv(DATA / dataset / "affinities.csv")
        grouped = measured.groupby("protein_id")["affinity"]
        reference = [grouped.get_group(p).to_numpy() if p in grouped.groups else np.array([]) for p in stats["scored_on"]]
        label = "training data, measured"

    positions = np.arange(len(stats), dtype=float)
    span = float(np.clip(0.34 * len(stats), 14, 30))

    figure, axes = plt.subplots(figsize=(span, 6))
    protein_stripes(axes, positions)
    draw_sides(reference, stats["values"], positions)

    axes.plot(positions, [np.mean(v) if len(v) else np.nan for v in reference], color=TRAINING_COLOUR, linewidth=2.0, label=label)

    axes.plot(positions, stats["mean"], color=GENERATED_COLOUR, linewidth=2.0, label="generated, predicted")

    axes.set_xlim(positions[0] - 0.75, positions[-1] + 0.75)
    axes.set_xlabel("Proteins (sorted by mean affinity)")
    axes.set_xticks(positions)
    axes.set_xticklabels([str(i) for i in range(len(stats))], fontsize=7)
    axes.set_ylabel("Affinity")
    if YLIM.get(dataset):
        axes.set_ylim(*YLIM[dataset])
    axes.legend(frameon=False)

    figure.tight_layout()

    path = output / "affinity.pdf"
    save_figure(figure, path, close=False)
    save_plot_data(path, stats.drop(columns="values"), dataset=dataset, branch=branch, oracle=ORACLE, run=run.name)
    plt.close(figure)

    return stats


def bootstrap_selectivity_ci(row, best_target, n_boot=N_BOOT, ci=0.95, rng=None):
    """e0's bootstrap: resample the off-targets, keep the best-target value fixed."""
    rng = np.random.default_rng(0) if rng is None else rng
    best = row[best_target]
    off = row.drop(best_target).to_numpy(dtype=float)

    boot = np.array([best - rng.choice(off, len(off), replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2

    return np.quantile(boot, [alpha, 1 - alpha])


def selectivity_heatmap(dataset, run, branch, output, top=TOP, seed=SEED):
    long, _ = load_predictions(run, branch)
    if long.empty:
        return None

    kept = long[long["counted"]] if COUNTED_ONLY else long
    wide = kept.pivot_table(index="molecule", columns="scored_on", values="affinity")
    if wide.shape[0] < 2 or wide.shape[1] < 2:
        print(f"{dataset:10s} {branch:5s} only {wide.shape[0]} molecules x {wide.shape[1]} targets after filtering -- no heatmap")
        return None

    labels = kept.drop_duplicates("molecule").set_index("molecule")["smiles"]
    designed = kept.drop_duplicates("molecule").set_index("molecule")["designed_for"]
    eligible = designed.isin(wide.columns)
    designed = designed[eligible]
    wide = wide.loc[designed.index]
    best_target = wide.idxmax(axis=1)
    selectivity = pd.Series({molecule: row[designed[molecule]] - row.drop(designed[molecule]).mean() for molecule, row in wide.iterrows()}, name="selectivity")

    candidates = pd.DataFrame({"designed_for": designed, "selectivity": selectivity})
    best_per_protein = candidates.groupby("designed_for")["selectivity"].idxmax()
    top_mols = candidates.loc[best_per_protein].nlargest(top, "selectivity").index.tolist()

    row_best_targets = [best_target[m] for m in top_mols]
    top_proteins = [designed[m] for m in top_mols]
    data = wide.loc[top_mols, top_proteins]

    data.index = [f"M{i:02d}" for i in range(1, len(top_mols) + 1)]
    data.index.name = "molecule"

    rng = np.random.default_rng(seed)
    cis = np.asarray([bootstrap_selectivity_ci(wide.loc[m], designed[m], rng=rng) for m in top_mols])
    sel = selectivity.loc[top_mols].to_numpy()
    xerr = np.vstack([sel - cis[:, 0], cis[:, 1] - sel])

    fig, (ax_heat, ax_sel) = plt.subplots(1, 2, figsize=(15, 10), sharey=True, gridspec_kw={"width_ratios": [4, 1], "wspace": -0.1})

    sns.heatmap(data, cmap="Spectral", cbar_kws={"label": "Affinity"}, ax=ax_heat, linewidths=0.5, linecolor="gray")

    for spine in ax_heat.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor("gray")

    ax_heat.set_xlabel("Protein ID")
    ax_heat.set_ylabel("Generated Molecule")
    ax_heat.set_yticklabels(data.index, rotation=0)
    ax_heat.set_xticklabels(data.columns, rotation=90)

    cbar = ax_heat.collections[0].colorbar
    cbar.set_label("")
    cbar.ax.text(0.5, 0.5, f"Predicted Affinity ({ORACLE})", rotation=270, ha="center", va="center", transform=cbar.ax.transAxes, fontweight="bold")

    y = np.arange(len(top_mols)) + 0.5

    ax_sel.errorbar(sel, y, xerr=xerr, fmt="o-", capsize=3)
    ax_sel.axvline(0, linestyle="--", linewidth=1)
    ax_sel.set_xlabel("Predicted Delta Score")
    ax_sel.set_ylabel("")

    for spine in ax_sel.spines.values():
        spine.set_visible(False)

    ax_sel.tick_params(axis="y", left=False, labelleft=False)
    ax_sel.annotate("",
        xy=(1.03, 0), xytext=(-0.03, 0),
        xycoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="->", linewidth=1.2), annotation_clip=False)

    path = output / "selectivity.pdf"
    save_figure(plt.gcf(), path, close=False)

    table = data.copy()
    table.columns = [f"protein_{c}" for c in table.columns]
    table.insert(0, "molecule_id", top_mols)
    table.insert(1, "designed_for", [designed[m] for m in top_mols])
    table["best_target"] = row_best_targets
    table["hit_designed_target"] = [designed[m] == best_target[m] for m in top_mols]
    table["delta_score"] = sel
    table["ci_low"], table["ci_high"] = cis[:, 0], cis[:, 1]
    table["smiles"] = [labels[m] for m in top_mols]
    save_plot_data(path, table.reset_index(), dataset=dataset, branch=branch, oracle=ORACLE)
    plt.close(fig)

    return table


def generated_sample(run, branch, per_target=PER_TARGET, seed=SEED):
    """Parseable generated attempts per target, capped equally at `per_target`."""
    frame = pd.read_csv(run / "generated.csv")
    frame = frame[(frame["branch"] == branch) & frame["smiles"].fillna("").ne("")]
    frame = frame[frame["smiles"].map(lambda text: Chem.MolFromSmiles(text) is not None)]
    frame = frame.sample(frac=1, random_state=seed)

    return frame.groupby("protein_id", group_keys=False).head(per_target)[["protein_id", "smiles"]]


def training_sample(dataset, targets, per_target=PER_TARGET):
    """Most potent measured ligands for each target, capped per target."""
    measured = pd.read_csv(DATA / dataset / "affinities.csv")
    measured = measured[measured["protein_id"].isin(targets)]
    measured = (measured.drop_duplicates(["protein_id", "drug_id"])
                        .sort_values(["protein_id", "affinity", "drug_id"], ascending=[True, False, True])
                        .groupby("protein_id", group_keys=False).head(per_target))

    drugs = pd.read_csv(DATA / dataset / "drugs.csv", index_col=0)
    frame = pd.DataFrame({"protein_id": measured["protein_id"].to_numpy(),
                        "smiles": drugs.loc[measured["drug_id"].to_numpy(),"smiles"].to_numpy()})

    return frame[frame["smiles"].fillna("").ne("")]


def distance_matrix(frame, targets):
    """(distances, labels, block edges) with rows grouped by target."""
    prints, labels = [], []
    for target in targets:                       # fixed order, so blocks are contiguous
        for text in frame.loc[frame["protein_id"] == target, "smiles"]:
            mol = Chem.MolFromSmiles(text)
            if mol is not None:
                prints.append(MORGAN.GetFingerprint(mol))
                labels.append(int(target))

    if len(prints) < 4:
        return None, None, None

    labels = np.asarray(labels)
    similarity = np.array([[1.0] * len(prints)] * len(prints), dtype=float)
    for i, fp in enumerate(prints):
        row = BulkTanimotoSimilarity(fp, prints[i + 1:])
        similarity[i, i + 1:] = row
        similarity[i + 1:, i] = row

    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)

    order = []
    for target in targets:
        rows = np.flatnonzero(labels == int(target))
        if len(rows) > 2:
            block = distance[np.ix_(rows, rows)]
            block = (block + block.T) / 2        # squareform wants exact symmetry
            np.fill_diagonal(block, 0.0)
            rows = rows[leaves_list(linkage(squareform(block, checks=False), method="average"))]
        order.extend(rows.tolist())

    order = np.asarray(order)
    ordered, labels = distance[np.ix_(order, order)], labels[order]
    edges = np.flatnonzero(np.diff(labels)) + 1

    return ordered, labels, edges


def block_scores(distance, labels):
    """(within, across, silhouette) on the very matrix the heatmap draws."""
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(distance, dtype=bool), k=1)

    within = 1.0 - distance[same & upper].mean() if (same & upper).any() else float("nan")
    across = 1.0 - distance[~same & upper].mean() if (~same & upper).any() else float("nan")

    quality = float("nan")
    if len(np.unique(labels)) > 1:
        quality = float(silhouette_score(distance, labels, metric="precomputed"))

    return float(within), float(across), quality


def draw(axes, distance, labels, edges):
    gap = 3
    block = np.searchsorted(edges, np.arange(len(labels)), side="right")
    mapped = np.arange(len(labels)) + gap * block
    expanded = np.full((len(labels) + gap * len(edges),) * 2, np.nan)
    expanded[np.ix_(mapped, mapped)] = 1.0 - distance

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#eeeeee")
    image = axes.imshow(expanded, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")

    bounds = np.concatenate([[0], edges, [len(labels)]])
    starts = bounds[:-1] + gap * np.arange(len(bounds) - 1)
    ends = bounds[1:] + gap * np.arange(len(bounds) - 1)
    centres = (starts + ends) / 2 - 0.5
    names = [str(labels[int(b)]) for b in bounds[:-1]]

    axes.set_xticks(centres)
    axes.set_xticklabels(names, rotation=90, fontsize=7)
    axes.set_yticks(centres)
    axes.set_yticklabels(names, fontsize=7)
    axes.set_xlabel("Protein")
    axes.set_ylabel("Protein")

    return image


def tanimoto_heatmap(dataset, run, branch, output, seed=SEED):
    generated = generated_sample(run, branch, seed=seed)
    candidates = generated["protein_id"].value_counts().index.to_numpy()
    represented = training_sample(dataset, candidates)["protein_id"].unique()
    candidates = np.intersect1d(candidates, represented)
    rng = np.random.default_rng(seed)
    targets = np.sort(rng.choice(candidates, min(MAX_TARGETS, len(candidates)), replace=False))
    generated = generated[generated["protein_id"].isin(targets)]

    if len(targets) < 2:
        print(f"{dataset:10s} {branch:5s} only {len(targets)} target(s) generated for, raise N_EVAL_TARGETS to block this")
        return None

    sources = [("training (control)", training_sample(dataset, targets)), (f"generated {branch}", generated)]

    panels = []
    for label, frame in sources:
        distance, labels, edges = distance_matrix(frame, targets)
        if distance is not None:
            panels.append((label, distance, labels, edges))

    if not panels:
        print(f"{dataset:10s} {branch:5s} too few parseable molecules to compare")
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 6.6), squeeze=False)
    rows = []

    for column, (label, distance, labels, edges) in enumerate(panels):
        within, across, quality = block_scores(distance, labels)
        image = draw(axes[0][column], distance, labels, edges)
        rows.append({"dataset": dataset, "branch": branch, "panel": label,
                        "molecules": len(labels), "targets": len(np.unique(labels)),
                        "block_min": int(pd.Series(labels).value_counts().min()),
                        "block_max": int(pd.Series(labels).value_counts().max()),
                        "tanimoto_within": within, "tanimoto_across": across,
                        "tanimoto_gap": within - across, "silhouette": quality})

    fig.colorbar(image, ax=axes[0].tolist(), label="Tanimoto similarity", fraction=0.025, pad=0.02)

    path = output / "tanimoto.pdf"
    save_plot_data(path, pd.DataFrame(rows))
    save_figure(fig, path)

    return pd.DataFrame(rows)

def generated_oracle_disagreement(run, branch="seq", counted_only=False):
    """One row per scored generated molecule with its on-target oracle disagreement."""
    store = np.load(run / "affinities.npz")
    panel = store["panel_ids"].astype(int)
    index = pd.read_csv(run / "scored_index.csv")
    index = index[index["branch"] == branch]

    frames = []
    for position in sorted(index["target"].unique()):
        members = pd.DataFrame({member: store[f"{branch}|{member}|{position}"][:, position].astype(float) for member in ORACLE_MEMBERS})
        rows = index[index["target"] == position].sort_values("row").reset_index(drop=True)
        if len(rows) != len(members):
            raise RuntimeError(f"{run.name} seq|{position}: scored_index and matrix disagree")
        frames.append(pd.concat([rows[["row", "smiles", "counted"]],
                            members.assign(mean_prediction=members.mean(axis=1),
                            oracle_disagreement=members.std(axis=1, ddof=0),
                            protein_id=int(panel[position]),
                            target_position=position)], axis=1))

    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True)
    return frame[frame["counted"]] if counted_only else frame


def similarity_to_training_reference(generated, reference):
    """Max Morgan Tanimoto to the reference set, one value per generated molecule."""
    fingerprints = [MORGAN.GetFingerprint(mol) for mol in reference if mol is not None]
    scores = []
    for mol in generated:
        if mol is None:
            scores.append(float("nan"))
            continue
        scores.append(float(max(BulkTanimotoSimilarity(
            MORGAN.GetFingerprint(mol), fingerprints))))

    return np.asarray(scores)


def _cluster_bootstrap(statistic, frame, cluster_column, draws, seed=0):
    """Percentile interval of `statistic(frame)` resampling whole clusters."""
    rng = np.random.default_rng(seed)
    groups = [block.index.to_numpy() for _, block in frame.groupby(cluster_column)]
    samples = []
    for _ in range(min(draws, ORACLE_BOOTSTRAPS)):
        picked = rng.integers(len(groups), size=len(groups))
        block = frame.loc[np.concatenate([groups[g] for g in picked])]
        samples.append(statistic(block))

    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def similarity_disagreement_statistics(frame, draws=ORACLE_BOOTSTRAPS):
    """Spearman association between similarity-to-reference and oracle disagreement."""
    rho, p_value = spearmanr(frame["max_tanimoto_to_reference"], frame["oracle_disagreement"])
    low, high = _cluster_bootstrap(lambda block: spearmanr(block["max_tanimoto_to_reference"], block["oracle_disagreement"]).statistic, frame, "target_position", draws)

    return pd.Series({
        "spearman_rho": float(rho), "spearman_p": float(p_value),
        "rho_ci_low": low, "rho_ci_high": high,
        "n_molecules": len(frame), "n_targets": frame["target_position"].nunique(),
        "disagreement_q1": frame["oracle_disagreement"].quantile(0.25),
        "disagreement_median": frame["oracle_disagreement"].median(),
        "disagreement_q4": frame["oracle_disagreement"].quantile(0.75),
    })


def heldout_disagreement_validation(frame, draws=ORACLE_BOOTSTRAPS):
    """Does disagreement track measured error on e1's held-out interactions?"""
    absolute = (frame["mean_prediction"] - frame["measured"]).abs()
    frame = frame.assign(absolute_error=absolute)
    rho, _ = spearmanr(frame["oracle_disagreement"], frame["absolute_error"])

    threshold = frame["absolute_error"].quantile(0.75)
    high = (frame["absolute_error"] >= threshold).astype(int)
    discriminated = high.nunique() == 2

    return pd.Series({
        "n_interactions": len(frame),
        "spearman_rho": float(rho),
        "mae_overall": float(absolute.mean()),
        "rmse_overall": float(np.sqrt((absolute ** 2).mean())),
        "auroc_high_error": (float(roc_auc_score(high, frame["oracle_disagreement"])) if discriminated else float("nan")),
        "average_precision": (float(average_precision_score(high, frame["oracle_disagreement"])) if discriminated else float("nan")),
        "mae_q4_over_q1": float(
            frame[frame["oracle_disagreement"] >= frame["oracle_disagreement"].quantile(0.75)]
            ["absolute_error"].mean()
            / frame[frame["oracle_disagreement"] <= frame["oracle_disagreement"].quantile(0.25)]
            ["absolute_error"].mean()),
    })


def compare_generated_reference(generated, reference, draws=ORACLE_BOOTSTRAPS):
    """Generated vs training-reference disagreement, clustered by panel target."""
    gen_medians = generated.groupby("target_position")["oracle_disagreement"].median()
    ref_medians = reference.groupby("target_position")["oracle_disagreement"].median()
    paired = pd.concat([gen_medians, ref_medians], axis=1, keys=("generated", "reference")).dropna()
    if paired.empty:
        return pd.Series(dtype=float)

    from scipy.stats import wilcoxon
    statistic, p_value = wilcoxon(paired["generated"], paired["reference"], alternative="greater")
    ratio = (paired["generated"] / paired["reference"]).median()

    return pd.Series({
        "n_targets": len(paired),
        "generated_median": paired["generated"].median(),
        "reference_median": paired["reference"].median(),
        "median_ratio": ratio,
        "wilcoxon_p_greater": float(p_value),
    })


def training_oracle_cache_path(dataset, seed):
    return (experiment_dir(dataset, "e7_ad_hoc") / "cache"
            / f"seed{seed}_training_oracle_scores.npz")


def training_baseline_cache(dataset, seed, panel_ids):
    """The cached training-molecule scores, or None if anything it keys on changed."""
    path = training_oracle_cache_path(dataset, seed)
    if not path.exists():
        return None

    with np.load(path, allow_pickle=False) as store:
        if (str(store["dataset"]) != dataset or int(store["seed"]) != seed
                or int(store["cache_version"]) != ORACLE_CACHE_VERSION
                or not np.array_equal(store["panel_ids"], np.asarray(panel_ids))
                or not np.array_equal(store["oracle_names"], np.array(list(ORACLE_MEMBERS)))):
            return None
        return {name: store[name] for name in store.files}


def score_training_oracle_baseline(dataset, seed, panel_ids, device):
    """Training molecules scored by every oracle member on the e6 panel, cached."""
    panel_ids = [int(p) for p in panel_ids]
    cached = training_baseline_cache(dataset, seed, panel_ids)
    if cached is not None:
        return cached

    from utils.data_tools import create_dataloaders, load_fingerprint_pca, load_tokenizer, shared_drug_ids
    from utils.experiment_tools import load_oracle, panel_loader, reference_molecules, score_with

    tokenizer = load_tokenizer(dataset, use_selfies=True)
    shared = shared_drug_ids(dataset, (load_tokenizer(dataset, use_selfies=False), tokenizer))
    (train_loader, _, test_loader), _ = create_dataloaders(
        dataset, tokenizer=tokenizer, batch_size=256, train_frac=0.8, val_frac=0.1,
        test_frac=0.1, random_state=seed, groups_per_batch=16, restrict_drug_ids=shared)

    reference = reference_molecules(train_loader, ORACLE_REFERENCE_MOLECULES, seed=seed)
    loader, kept = panel_loader(reference, panel_ids, test_loader, tokenizer, fingerprint_pca=load_fingerprint_pca(dataset), device=device)
    scores = np.stack([score_with(loader, panel_ids, load_oracle(dataset, member, device=device), test_loader, device).numpy()
                        for member in ORACLE_MEMBERS]).astype(np.float32)

    # panel_loader's second return is already canonical SMILES strings, not molecules
    smiles = np.array([text for text in kept[: scores.shape[1]]])
    path = training_oracle_cache_path(dataset, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, scores=scores, oracle_names=np.array(ORACLE_MEMBERS),
                        smiles=smiles, panel_ids=np.asarray(panel_ids),
                        dataset=np.array(dataset), seed=np.array(seed),
                        sample_requested=np.array(ORACLE_REFERENCE_MOLECULES),
                        cache_version=np.array(ORACLE_CACHE_VERSION))
    temporary.replace(path)

    return training_baseline_cache(dataset, seed, panel_ids)


def oracle_validation_frame(dataset, device):
    """Three oracle predictions against measured affinity on e1's test split."""
    import torch
    from utils.data_tools import create_dataloaders
    from utils.experiment_tools import load_oracle

    cache = (experiment_dir(dataset, "e7_ad_hoc") / "cache"
                / f"heldout_oracle_predictions_v{ORACLE_CACHE_VERSION}.csv")
    if cache.exists():
        return pd.read_csv(cache)

    (_, _, test_loader), (_, _, test_table) = create_dataloaders(
        dataset, batch_size=1024, train_frac=0.8, val_frac=0.1, test_frac=0.1,
        cold_protein=False, random_state=SPLIT_SEED_E1, device=device)

    predictions, measured = [], test_table["affinity"].to_numpy(dtype=float)
    for member in ORACLE_MEMBERS:
        oracle = load_oracle(dataset, member, device=device).eval()
        values = []
        with torch.no_grad():
            for batch in test_loader:
                values.append(oracle(batch).view(-1).cpu().numpy())
        predictions.append(np.concatenate(values))
        del oracle
        torch.cuda.empty_cache()

    stacked = np.stack(predictions, axis=1)
    frame = pd.DataFrame({
        "drug_id": test_table["drug_id"].to_numpy(),
        "protein_id": test_table["protein_id"].to_numpy(),
        "measured": measured,
        **{member: stacked[:, i] for i, member in enumerate(ORACLE_MEMBERS)},
        "mean_prediction": stacked.mean(axis=1),
        "oracle_disagreement": stacked.std(axis=1, ddof=0),
    })

    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)

    return frame


def _binned_interval(grouped, statistic, cluster_column, draws=BIN_BOOTSTRAPS):
    """95% cluster-bootstrap interval on `statistic`, one per bin."""
    low, high = [], []
    for _, block in grouped:
        try:
            bounds = _cluster_bootstrap(statistic, block, cluster_column, draws)
        except (ValueError, IndexError):        # a bin holding a single cluster
            value = statistic(block)
            bounds = (value, value)
        low.append(bounds[0])
        high.append(bounds[1])

    return low, high


def oracle_disagreement_figure(generated, reference, validation, path, units):
    """The three things the disagreement analysis has to say, side by side."""
    figure, cells = plt.subplots(1, 3, figsize=(15, 4.6))

    cells[0].boxplot([reference["oracle_disagreement"].to_numpy(),
                        generated["oracle_disagreement"].to_numpy()],
                        positions=[0, 1], widths=0.6, vert=False, patch_artist=True,
                        showfliers=False,
                        medianprops={"color": "#11181f", "linewidth": 1.6},
                        boxprops={"facecolor": "#cfd8e3", "edgecolor": "#67727e"})
    cells[0].set_yticks([0, 1])
    cells[0].set_yticklabels(["Training\nmolecules", "Generated\nmolecules"])

    pairs = generated.dropna(subset=["max_tanimoto_to_reference"])
    bands = pd.qcut(pairs["oracle_disagreement"], ORACLE_BINS, duplicates="drop")
    grouped = pairs.groupby(bands, observed=True)
    centres = grouped["oracle_disagreement"].mean()
    cells[1].plot(centres, grouped["max_tanimoto_to_reference"].mean(), marker="o", color="#1d7874")
    low, high = _binned_interval(grouped, lambda block: float(block["max_tanimoto_to_reference"].mean()), "target_position")
    cells[1].fill_between(centres, low, high, color="#1d7874", alpha=0.18)
    cells[1].set_ylabel("Tanimoto to nearest training molecule")
    rho, _ = spearmanr(pairs["max_tanimoto_to_reference"], pairs["oracle_disagreement"])
    cells[1].annotate(f"Spearman $\\rho$ = {rho:+.3f}", xy=(0.04, 0.06), xycoords="axes fraction", fontsize=11)

    if validation is not None and not validation.empty:
        errors = validation.assign(absolute_error=(validation["mean_prediction"] - validation["measured"]).abs())
        bins = pd.qcut(errors["oracle_disagreement"], ORACLE_BINS, duplicates="drop")
        by_bin = errors.groupby(bins, observed=True)
        levels = by_bin["oracle_disagreement"].mean()
        rmse = by_bin["absolute_error"].apply(
            lambda block: float(np.sqrt((block ** 2).mean())))
        low, high = _binned_interval(
            by_bin,
            lambda block: float(np.sqrt((block["absolute_error"] ** 2).mean())),
            "protein_id")
        cells[2].plot(levels, rmse, marker="o", color="#b5485d")
        cells[2].fill_between(levels, low, high, color="#b5485d", alpha=0.18)
        cells[2].set_ylabel("RMSE")
        error_rho, _ = spearmanr(errors["oracle_disagreement"], errors["absolute_error"])
        cells[2].annotate(f"Spearman $\\rho$ = {error_rho:+.3f}", xy=(0.04, 0.93), xycoords="axes fraction", fontsize=11)
    else:
        cells[2].set_visible(False)

    for cell in cells:
        cell.set_xlabel("Oracle disagreement")

    figure.tight_layout()

    save_plot_data(path, pairs[["oracle_disagreement", "max_tanimoto_to_reference"]], affinity_units=units)
    save_figure(figure, path)

    return path


def oracle_disagreement_analysis(dataset, run, output, device, validation=None):
    """Generated vs training-reference disagreement, plus the similarity association."""
    generated = generated_oracle_disagreement(run, branch="seq", counted_only=True)
    if generated.empty:
        return None

    panel = np.load(run / "affinities.npz")["panel_ids"].astype(int)
    baseline = score_training_oracle_baseline(dataset, run_seed(run), list(panel), device)
    reference = pd.DataFrame(
        baseline["scores"].std(axis=0, ddof=0).ravel(),
        columns=["oracle_disagreement"]).assign(
        target_position=np.tile(np.arange(len(panel)), baseline["scores"].shape[1]))

    # the cached reference smiles are already canonical strings
    reference_mols = [Chem.MolFromSmiles(text) for text in baseline["smiles"]]
    generated["max_tanimoto_to_reference"] = similarity_to_training_reference([Chem.MolFromSmiles(text) for text in generated["smiles"]], reference_mols)

    rows = {"dataset": dataset, "seed": run_seed(run), "run": run.name,
            "writeup_seed": run_seed(run) == WRITEUP_SEED,
            **compare_generated_reference(generated, reference),
            **similarity_disagreement_statistics(generated.dropna(subset=["max_tanimoto_to_reference"]))}

    table = pd.DataFrame([rows])
    table.to_csv(output / "oracle_disagreement.csv", index=False)
    oracle_disagreement_figure(generated, reference, validation,
                                output / "oracle_disagreement_panels.pdf",
                                affinity_units(dataset))
    print(f"  oracle disagreement: generated/reference median ratio {rows['median_ratio']:.2f}x over {rows['n_targets']} targets")

    return table


def main():
    device = setup_compute()

    for dataset in DATASETS:
        runs = [run for run in find_runs(dataset) if run_seed(run) in ORACLE_SEEDS]
        if not runs:
            print(f"{dataset:10s} no {SOURCE} run with a {MODEL} library yet -> skipped")
            continue

        # once per dataset
        measured = None
        try:
            measured = oracle_validation_frame(dataset, device)
            validation = heldout_disagreement_validation(measured)
            frame = validation.to_frame("value").reset_index().rename(columns={0: "metric"})
            experiment_dir(dataset, "e7_ad_hoc").mkdir(parents=True, exist_ok=True)
            frame.to_csv(experiment_dir(dataset, "e7_ad_hoc") / "oracle_validation.csv", index=False)
            print(f"{dataset:10s} oracle validation: rho "
                    f"{validation['spearman_rho']:+.3f}, MAE Q4/Q1 "
                    f"{validation['mae_q4_over_q1']:.2f}x")
        except FileNotFoundError as missing:
            print(f"{dataset:10s} oracle validation skipped, {missing}")

        for run in runs:
            output = figure_dir(dataset, run)
            if "seq" not in available_branches(run):
                continue

            stats = affinity_per_protein(dataset, run, "seq", output)
            if stats is None:
                continue

            print(f"{dataset:10s} {run.name:22s} seq   {len(stats)} proteins, "
                    f"mean {stats['mean'].min():.2f}-{stats['mean'].max():.2f}, "
                    f"{int(stats['n'].sum()):,} molecules")

            selectivity_heatmap(dataset, run, "seq", output)
            tanimoto_heatmap(dataset, run, "seq", output, seed=run_seed(run))

            try:
                oracle_disagreement_analysis(dataset, run, output, device, validation=measured)
            except FileNotFoundError as missing:
                print(f"  oracle disagreement skipped, {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
