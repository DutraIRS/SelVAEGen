import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from scipy import stats

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "experiments"))

import e6_compare_models as source_module
from utils import DATA, save_figure, save_plot_data
from utils.data_tools import (PROT_EMB_MODEL, create_dataloaders, load_fingerprint_pca, load_protein_embeddings, load_tokenizer, shared_drug_ids)
from utils.experiment_tools import (MeanOracle, cold_targets, collect_runs,
                                    experiment_dir, generate_libraries, load_oracle,
                                    load_weights, max_atoms, measured_no_bond_weight,
                                    record_run, reference_molecules, run_dir,
                                    save_config, score_libraries, setup_compute,
                                    training_smiles)
from utils.metrics import Metrics
from utils.stats_tools import benjamini_hochberg

EXPERIMENT = "e9_warm_panel"
SOURCE = "e6_compare"
DATASETS = ["kiba", "bindingdb", "papyrus", "davis"]
MODELS = ["SelVAEGen"]
SEEDS = list(source_module.SEEDS)

COLD_BANDS = 4                     # similarity quartiles over the cold test targets
TARGETS_PER_BAND = None            # None levels every band to the smallest one offered
OFF_TARGETS = {"kiba": 3, "bindingdb": 8, "papyrus": 8, "davis": 8}
INHERITED = ("BATCH_SIZE", "GROUPS_PER_BATCH", "N_GENERATED", "N_REFERENCE",
                "SUCCESS_THRESHOLD", "REPORTING_ORACLES", "SELVAEGEN_USE_SELFIES")

GRADIENT_METRICS = ["seq_MeanOracle_delta_score",
                    "seq_MeanOracle_directional_consistency",
                    "seq_MeanOracle_effective_delta"]

# column, axis label, value under no conditioning
GRADIENT_PANELS = (
    ("seq_MeanOracle_delta_score", "$\\Delta$ score", 0.0),
    ("seq_MeanOracle_directional_consistency", "Directional consistency", 0.5),
    ("seq_MeanOracle_effective_delta", "Effective $\\Delta$", 0.0),
)
BAND_COLOUR = "#2f6fb5"
WARM_COLOUR = "#b5485d"

def max_train_similarity(embeddings, train_ids, candidate_ids):
    """Maximum cosine similarity and nearest training protein per candidate."""
    train_ids = [int(value) for value in train_ids]
    candidate_ids = [int(value) for value in candidate_ids]

    train = torch.nn.functional.normalize(embeddings[train_ids].float(), dim=1)
    candidates = torch.nn.functional.normalize(embeddings[candidate_ids].float(), dim=1)
    values, positions = (candidates @ train.T).max(dim=1)

    return pd.DataFrame({
        "protein_id": candidate_ids,
        "nearest_train_id": [train_ids[int(position)] for position in positions],
        "similarity": values.cpu().numpy(),
    })


def load_identity(dataset):
    """The percent-identity matrix scripts/run_blast.py wrote, and where each protein sits."""
    with np.load(DATA / dataset / "protein_identity.npz") as saved:
        return (saved["identity"], {int(protein): index for index, protein in enumerate(saved["protein_ids"])})


def max_train_identity(identity, position, train_ids, candidate_ids):
    """Highest percent identity to any training protein, per candidate."""
    rows = [position[int(value)] for value in candidate_ids]
    columns = [position[int(value)] for value in train_ids]
    block = identity[np.ix_(rows, columns)]
    best = block.argmax(axis=1)

    return pd.DataFrame({
        "protein_id": [int(value) for value in candidate_ids],
        "nearest_train_id": [int(train_ids[index]) for index in best],
        "similarity": block[np.arange(len(rows)), best],
    })


def similarity_bands(frame, cold_bands=4, targets_per_band=None, seed=0):
    """Rank-balanced cold bands plus warm proteins as the similarity-one anchor."""
    rng = np.random.default_rng(seed)
    
    cold = frame[frame["panel"] == "cold"].sort_values(["similarity", "protein_id"]).reset_index(drop=True)
    warm = frame[frame["panel"] == "warm"].sort_values("protein_id").reset_index(drop=True)
    
    groups = np.array_split(np.arange(len(cold)), min(cold_bands, len(cold))) if len(cold) else []
    
    if targets_per_band is None:
        targets_per_band = min((len(group) for group in groups), default=len(warm))

    selected = []
    for index, positions in enumerate(groups):
        block = cold.iloc[positions].copy()
        if len(block) > targets_per_band:
            picked = np.sort(rng.choice(len(block), targets_per_band, replace=False))
            block = block.iloc[picked].copy()
        
        block["band"] = f"cold_q{index + 1}"
        block["band_index"] = index
        selected.append(block)

    if not warm.empty:
        if len(warm) > targets_per_band:
            picked = np.sort(rng.choice(len(warm), targets_per_band, replace=False))
            warm = warm.iloc[picked].copy()
        
        warm["band"] = "warm"
        warm["band_index"] = len(groups)
        selected.append(warm)

    return pd.concat(selected, ignore_index=True).sort_values(["band_index", "similarity", "protein_id"]).reset_index(drop=True)


def fixed_off_target_panel(target, off_targets):
    """Conditioning target first, followed by the reserved off-target panel."""
    return [int(target)] + [int(value) for value in off_targets]


def retrieval_metrics(generated_smiles, target_smiles, training_smiles):
    """Exact target-conditioned retrieval against its training-set base rate."""
    def canonical(values):
        kept = set()
        for text in values:
            mol = Chem.MolFromSmiles(text) if isinstance(text, str) and text else None
            if mol is not None:
                kept.add(Chem.MolToSmiles(mol, canonical=True))
        return kept

    generated = canonical(generated_smiles)
    target = canonical(target_smiles)
    training = canonical(training_smiles)

    return {"retrieval_precision": len(generated & target) / len(generated),
            "retrieval_base_rate": len(target & training) / len(training),
            "retrieval_unique": len(generated)}


def top1_recovery(affinities, target_index=0):
    """Fraction whose highest predicted affinity is the conditioning target."""
    values = np.asarray(affinities, dtype=float)

    return float((values.argmax(axis=1) == int(target_index)).mean())


def blocked_gradient(frame, metrics):
    """Per-seed slopes followed by a one-sample test across seed blocks."""
    rows = []
    for (seed, model), block in frame.groupby(["seed", "model"]):
        for metric in metrics:
            pair = block[["similarity", metric]].dropna()
            slope = np.polyfit(pair["similarity"], pair[metric], 1)[0]
            rows.append({"seed": seed, "model": model, "metric": metric, "slope": float(slope), "bands": len(pair)})

    slopes = pd.DataFrame(rows)
    tests = []
    if slopes.empty:
        return slopes, pd.DataFrame()

    for (model, metric), block in slopes.groupby(["model", "metric"]):
        values = block["slope"].to_numpy(dtype=float)
        mean = float(values.mean())
        spread = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        if len(values) < 2:
            statistic = pvalue = float("nan")
        elif np.isclose(spread, 0.0):
            statistic, pvalue = ((float("inf"), 0.0) if mean else (0.0, 1.0))
        else:
            result = stats.ttest_1samp(values, 0.0)
            statistic, pvalue = float(result.statistic), float(result.pvalue)
        tests.append({"model": model, "metric": metric, "n": len(values),
                    "mean_slope": mean, "sd_slope": spread,
                    "t": statistic, "p": pvalue})

    tests = pd.DataFrame(tests)
    tests["p_BH"], tests["significant"] = benjamini_hochberg(tests["p"])
    return slopes, tests.sort_values(["model", "p"]).reset_index(drop=True)


def summarize_target_scores(scores):
    """One band row from equally weighted per-target score dictionaries."""
    numeric = pd.DataFrame(scores).select_dtypes(include=[np.number])
    result = {}
    for column in numeric:
        if column.endswith("_n_targets") or column.endswith("_count"):
            result[column] = float(numeric[column].sum())
        else:
            result[column] = float(numeric[column].mean())

    base = result["retrieval_base_rate"]
    result["retrieval_lift"] = (result["retrieval_precision"] / base if base else float("nan"))
    return result


def gradient_figure(frame, present, path):
    """Every pre-specified endpoint against protein similarity, one panel each."""
    from matplotlib import pyplot as plt

    bands = (frame.groupby(["band_index", "band"], as_index=False)["similarity"].mean().sort_values("band_index"))
    order = list(bands["band"])
    x = bands["similarity"].to_numpy(dtype=float)
    panels = [panel for panel in GRADIENT_PANELS if panel[0] in present]

    columns = 2 if len(panels) == 4 else min(len(panels), 3)
    rows = int(np.ceil(len(panels) / columns))
    figure, cells = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.1 * rows), squeeze=False)
    cells = cells.ravel()

    for cell, (column, label, null) in zip(cells, panels):
        grouped = frame.groupby("band")[column]
        mean = grouped.mean().reindex(order).to_numpy(dtype=float)

        for _, block in frame.groupby("seed"):
            values = block.set_index("band")[column].reindex(order).to_numpy(dtype=float)
            cell.plot(x, values, color=BAND_COLOUR, alpha=0.30, linewidth=0.9, marker="o", markersize=3, zorder=2)

        cell.plot(x, mean, color=BAND_COLOUR, linewidth=2.0, marker="o", markersize=6, zorder=3)
        
        cell.scatter(x[-1:], mean[-1:], s=70, color=WARM_COLOUR, zorder=4, label="Warm (training proteins)")

        if null is not None:
            cell.axhline(null, color="#b0392b", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)
        cell.set_ylabel(label)
        cell.invert_xaxis()

    for cell in cells[len(panels):]:
        cell.set_visible(False)
    
    for cell in cells[max(0, len(panels) - columns):len(panels)]:
        cell.set_xlabel("Identity to nearest training protein (%)")

    cells[0].legend(fontsize=11, frameon=False, loc="best")
    figure.tight_layout()

    save_plot_data(path, frame[["seed", "band", "band_index", "similarity"]
                                + [panel[0] for panel in panels]])
    save_figure(figure, path)
    plt.close(figure)

    return path


def write_analysis(frame, output):
    """Band table, per-seed gradient slopes with their blocked test, and summary."""
    if frame.empty:
        return

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # similarity is averaged
    present = [name for name in GRADIENT_METRICS if name in frame]
    band_means = (frame.groupby(["model", "band", "band_index"], as_index=False)
                    [present + ["similarity"]].mean()
                    .sort_values(["model", "band_index"]))
    band_means.to_csv(output / "gradient.csv", index=False)

    slopes, tests = blocked_gradient(frame, present)
    slopes.to_csv(output / "gradient_slopes.csv", index=False)
    tests.to_csv(output / "gradient_tests.csv", index=False)

    columns = [c for c in frame.columns if c.startswith(("seq_", "graph_")) and not c.endswith(("_n", "_n_targets"))]
    summary = (frame.groupby(["model", "band"])[columns + ["similarity"]]
                .agg(["mean", "std", "count"])
                .stack(level=0, future_stack=True).reset_index()
                .rename(columns={"level_2": "metric"}))
    summary.to_csv(output / "summary.csv", index=False)

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    gradient_figure(frame, present, figures / "similarity_gradient.pdf")

    if not tests.empty:
        print(f"\n{'=' * 78}\nsimilarity gradient, per-seed slopes tested across seeds")
        print(f"{'=' * 78}")
        print(tests.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


def recorded_cold_panel(checkpoint):
    """The cold panel the source experiment actually scored this checkpoint against."""
    with np.load(Path(checkpoint).parent / "affinities.npz") as saved:
        return [int(value) for value in saved["panel_ids"]]


def band_panel(dataset, seed, test_loader, train_table, recorded):
    """One plan per band: its conditioning targets and the panel they are scored against."""
    cold = cold_targets(test_loader)
    train_ids = sorted(int(p) for p in train_table["protein_id"].unique())
    warm_pool = [p for p in train_ids if p not in set(cold)]

    # the split is rebuilt from random_state, so check it against what the source recorded
    if set(recorded) != set(cold):
        raise RuntimeError(
            f"{dataset} seed{seed}: the rebuilt cold panel is not the one {SOURCE} scored "
            f"this checkpoint against -- {len(set(recorded) & set(cold))} of "
            f"{len(recorded)} shared; the split inputs have drifted since {SOURCE} ran")

    candidates = cold + warm_pool
    identity, position = load_identity(dataset)
    similarities = max_train_identity(identity, position, train_ids, candidates)

    # the cosine is what the model conditions on; the bands are cut on identity
    embeddings = load_protein_embeddings(dataset, prot_emb_model=PROT_EMB_MODEL)
    cosine = max_train_similarity(embeddings, train_ids, candidates)
    similarities["embedding_cosine"] = cosine["similarity"].to_numpy()
    similarities["panel"] = ["warm" if p in set(warm_pool) else "cold" for p in similarities["protein_id"]]
    return band_plans(similarities, cold, seed, OFF_TARGETS[dataset])


def band_plans(similarities, cold, seed, off_targets):
    """One plan per band, holding every conditioning target that band contributes."""
    # cold_targets orders by measurement count
    off_targets = [int(value) for value in cold[:off_targets]]
    conditioners = similarities[~similarities["protein_id"].isin(off_targets)]

    selected = similarity_bands(conditioners, cold_bands=COLD_BANDS, targets_per_band=TARGETS_PER_BAND, seed=seed)

    plans = []
    for (band, band_index), block in selected.groupby(["band", "band_index"], sort=True):
        plans.append({
            "band": str(band), "band_index": int(band_index),
            "similarity": float(block["similarity"].mean()),
            "embedding_cosine": float(block["embedding_cosine"].mean()),
            "targets": [{"protein_id": int(row.protein_id),
                            "similarity": float(row.similarity),
                            "panel_ids": fixed_off_target_panel(row.protein_id, off_targets)}
                        for row in block.itertuples(index=False)]})

    return sorted(plans, key=lambda plan: plan["band_index"])


def run_dataset(dataset, device, models=None, seeds=None, analyse=True):
    """Every (seed, model, band) for one dataset, resuming what is already done."""
    RDLogger.DisableLog("rdApp.*")
    models = MODELS if models is None else models
    seeds = SEEDS if seeds is None else seeds

    source = experiment_dir(dataset, SOURCE)
    output = experiment_dir(dataset, EXPERIMENT)

    config_path = source / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} is missing; run {SOURCE} first")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    inherited = {name: config.get(name, getattr(source_module, name, None)) for name in INHERITED}
    save_config(output, sys.modules[__name__], source_experiment=SOURCE, source_commit=config.get("commit"), **inherited)

    records, finished = collect_runs(output, ("seed", "model", "band"), valid={"seed": seeds})

    metrics = Metrics()
    use_selfies = bool(inherited["SELVAEGEN_USE_SELFIES"])
    tokenizer = load_tokenizer(dataset, use_selfies=use_selfies)
    
    # both tokenizers
    shared = shared_drug_ids(dataset, (load_tokenizer(dataset, use_selfies=True), load_tokenizer(dataset, use_selfies=False)))
    fingerprint_pca = load_fingerprint_pca(dataset)

    names = inherited["REPORTING_ORACLES"] or ["GraphDTAOracle", "GSDTAOracle", "BaselineOracle"]
    
    oracles = {name: load_oracle(dataset, name, device=device) for name in names}
    oracles["MeanOracle"] = MeanOracle(list(oracles.values())).to(device).eval()

    # index_col=0 everywhere
    drugs = pd.read_csv(DATA / dataset / "drugs.csv", index_col=0)
    interactions = pd.read_csv(DATA / dataset / "affinities.csv")
    threshold = inherited["SUCCESS_THRESHOLD"][dataset]

    started, written = time.time(), 0
    for seed in seeds:
        for name in models:
            checkpoint = source / "runs" / f"seed{seed}_{name}" / "model.pt"
            if not checkpoint.exists():
                print(f"  {dataset} seed{seed} {name}: no checkpoint, skipped", flush=True)
                continue

            torch.manual_seed(seed)
            (train_loader, val_loader, test_loader), (train_table, val_table, _) = \
                create_dataloaders(
                    dataset, tokenizer=tokenizer,
                    batch_size=inherited["BATCH_SIZE"], train_frac=0.8, val_frac=0.1,
                    test_frac=0.1, random_state=seed, device=device,
                    groups_per_batch=inherited["GROUPS_PER_BATCH"],
                    restrict_drug_ids=shared)

            plans = band_panel(dataset, seed, test_loader, train_table, recorded_cold_panel(checkpoint))
            todo = [plan for plan in plans if not finished(seed=seed, model=name, band=plan["band"])]

            print(f"\n{dataset} seed{seed} {name}: {len(todo)} band(s) to do", flush=True)
            if not todo:
                del train_loader, val_loader, test_loader
                torch.cuda.empty_cache()
                continue

            reference = reference_molecules(train_loader, inherited["N_REFERENCE"])
            known = training_smiles(train_loader)
            activations = metrics.chemnet_activations([Chem.MolToSmiles(mol) for mol in reference if mol is not None])
            nodes = max_atoms(train_loader, val_loader)
            no_bond_weight = measured_no_bond_weight(train_loader)

            model = load_weights(
                source_module.build_model(
                    name, tokenizer, train_loader, train_table, val_table, nodes, seed,
                    device, no_bond_weight=no_bond_weight).to(device),
                checkpoint, device)

            for plan in todo:
                directory = run_dir(dataset, EXPERIMENT, f"seed{seed}_{name}_{plan['band']}")
                collected = []

                for target in plan["targets"]:
                    panel = target["panel_ids"]
                    
                    libraries = generate_libraries(
                        model, test_loader, panel[:1],
                        n_generated=inherited["N_GENERATED"], seed=seed, device=device,
                        size_loader=train_loader)

                    here = directory / f"target{target['protein_id']}"
                    scores = score_libraries(
                        libraries, tokenizer, test_loader, metrics, oracles, panel,
                        reference, fingerprint_pca=fingerprint_pca,
                        success_threshold=inherited["SUCCESS_THRESHOLD"][dataset],
                        reference_activations=activations, device=device,
                        save_to=here, known_smiles=known)

                    with np.load(here / "affinities.npz") as store:
                        scores["top1_recovery"] = top1_recovery(store["seq|MeanOracle|0"], target_index=0)

                    binders = interactions.loc[
                        (interactions["protein_id"] == target["protein_id"])
                        & (interactions["affinity"] >= threshold), "drug_id"]
                    generated = pd.read_csv(here / "generated.csv")
                    scores.update(retrieval_metrics(
                        generated.loc[generated["branch"] == "seq", "smiles"].tolist(),
                        set(drugs.loc[binders.tolist(), "smiles"]), known))
                    collected.append(scores)

                if not collected:
                    continue

                row = {"dataset": dataset, "seed": seed, "model": name,
                        "band": plan["band"], "band_index": plan["band_index"],
                        "similarity": plan["similarity"],
                        "conditioners": len(collected),
                        "evaluation_split": "train" if plan["band"] == "warm" else "test",
                        "n_panel": len(plan["targets"][0]["panel_ids"]),
                        **summarize_target_scores(collected)}
                record_run(directory, row)
                records, finished = collect_runs(output, ("seed", "model", "band"), valid={"seed": seeds})
                written += 1
                print(f"  {plan['band']:8s} sim {plan['similarity']:.3f} "
                        f"n={len(collected):<3d} "
                        f"delta {row.get('seq_MeanOracle_delta_score', float('nan')):+.4f}  "
                        f"dircon {row.get('seq_MeanOracle_directional_consistency', float('nan')):.4f}  "
                        f"top1 {row.get('top1_recovery', float('nan')):.3f}",
                        flush=True)

            del model
            torch.cuda.empty_cache()
            del train_loader, val_loader, test_loader
            torch.cuda.empty_cache()

    frame = pd.DataFrame(records)
    print(f"\n{dataset}: {written} new scoring pass(es) in {(time.time() - started) / 60:.1f} min, {len(frame)} rows total")

    if analyse and not frame.empty:
        write_analysis(frame, output)

    return frame


def main():
    device = setup_compute()
    frames = []
    for dataset in DATASETS:
        print(chr(10) + "=" * 70 + chr(10) + f"=== {dataset}" + chr(10) + "=" * 70, flush=True)
        try:
            frames.append(run_dataset(dataset, device))
        except FileNotFoundError as missing:
            print(f"  {dataset}: skipped, {missing}", flush=True)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    frame = main()

    if frame.empty:
        raise SystemExit(
            f"no dataset produced any rows. {EXPERIMENT} reloads "
            f"results/<dataset>/{SOURCE}/runs/*/model.pt, which finish_run writes only "
            f"when a run completes -- so {SOURCE} has to have finished, with its model.pt "
            f"files kept.")
