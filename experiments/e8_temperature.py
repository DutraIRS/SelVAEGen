import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "experiments"))

import e6_compare_models as source_module
from utils.data_tools import create_dataloaders, load_fingerprint_pca, load_tokenizer, shared_drug_ids
from utils.experiment_tools import (MeanOracle, cold_targets, collect_runs,
                                    experiment_dir, generate_libraries, load_oracle,
                                    load_weights, max_atoms, measured_no_bond_weight,
                                    plot_metric_curves, record_run, reference_molecules,
                                    run_dir, save_config, score_libraries,
                                    setup_compute, training_smiles)
from utils.metrics import Metrics

EXPERIMENT = "e8_temperature"
SOURCE = "e6_compare"

DATASETS = ["kiba", "bindingdb", "papyrus", "davis"]
MODELS = ["SelVAEGen"]
ARM_ORDER = list(source_module.MODELS) 
SEEDS = list(source_module.SEEDS)

TEMPERATURES = [round(0.25 * step, 2) for step in range(21)]
REFERENCE_TEMPERATURE = 1.0        # where every other experiment generated

INHERITED = ("BATCH_SIZE", "GROUPS_PER_BATCH", "N_GENERATED", "N_REFERENCE",
                "SUCCESS_THRESHOLD", "REPORTING_ORACLES", "SELVAEGEN_USE_SELFIES")

BASELINES = ("DeepDTAGen", "Prot2Drug", "ZeroGEN")
WATCH = ["uniqueness", "novelty", "MeanOracle_effective_delta",
            "MeanOracle_delta_score", "MeanOracle_directional_consistency", "tanimoto_gap"]
KNEE_METRICS = ["MeanOracle_effective_delta", "MeanOracle_delta_score",
                "MeanOracle_directional_consistency", "novelty", "uniqueness"]

CONTROL_TOLERANCE = 0.01           # T = 1 must reproduce the source run this closely


def tokenizer_kind(name, use_selfies):
    """Which tokenizer this model's checkpoint was trained with."""
    return "smiles" if name in BASELINES else ("selfies" if use_selfies else "smiles")


def run_dataset(dataset, device, models, temperatures, seeds, analyse=True):
    """Every (seed, model, temperature) for one dataset, resuming what is already done."""
    source = experiment_dir(dataset, SOURCE)
    output = experiment_dir(dataset, EXPERIMENT)

    config_path = source / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} is missing; run {SOURCE} first")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    inherited = {name: config.get(name, getattr(source_module, name, None)) for name in INHERITED}

    save_config(output, sys.modules[__name__], source_experiment=SOURCE, source_commit=config.get("commit"), **inherited)

    grid = {"seed": sorted(set(SEEDS) | set(seeds)),
            "model": list(dict.fromkeys(ARM_ORDER + list(models))),
            "temperature": sorted(set(TEMPERATURES) | set(temperatures))}

    records, finished = collect_runs(output, ("seed", "model", "temperature"), valid=grid)

    metrics = Metrics()
    use_selfies = bool(inherited["SELVAEGEN_USE_SELFIES"])
    tokenizers = {"selfies": load_tokenizer(dataset, use_selfies=True),
                    "smiles": load_tokenizer(dataset, use_selfies=False)}
    shared = shared_drug_ids(dataset, tokenizers.values())
    fingerprint_pca = load_fingerprint_pca(dataset)

    names = inherited["REPORTING_ORACLES"] or ["GraphDTAOracle", "GSDTAOracle", "BaselineOracle"]
    oracles = {name: load_oracle(dataset, name, device=device) for name in names}
    oracles["MeanOracle"] = MeanOracle(list(oracles.values())).to(device).eval()

    started = time.time()
    written = 0

    for seed in seeds:
        # one loader per (seed, tokenizer)
        for kind in ("selfies", "smiles"):
            wanted = [name for name in models if tokenizer_kind(name, use_selfies) == kind]
            here = [name for name in wanted if (source / "runs" / f"seed{seed}_{name}" / "model.pt").exists()]

            for name in wanted:
                if name not in here:
                    print(f"  {dataset} seed{seed} {name}: no checkpoint at "
                            f"{source.name}/runs/seed{seed}_{name}/model.pt, skipped",
                            flush=True)

            todo = {name: [t for t in temperatures if not finished(seed=seed, model=name, temperature=t)] for name in here}
            here = [name for name in here if todo[name]]
            if not here:
                continue

            tokenizer = tokenizers[kind]
            torch.manual_seed(seed)
            (train_loader, val_loader, test_loader), (train_table, val_table, _) = \
                create_dataloaders(
                    dataset, tokenizer=tokenizer,
                    batch_size=inherited["BATCH_SIZE"], train_frac=0.8, val_frac=0.1,
                    test_frac=0.1, random_state=seed, device=device,
                    groups_per_batch=inherited["GROUPS_PER_BATCH"],
                    restrict_drug_ids=shared)

            panel = cold_targets(test_loader)          # the panel the run was scored on
            reference = reference_molecules(train_loader, inherited["N_REFERENCE"])
            known = training_smiles(train_loader)
            activations = metrics.chemnet_activations([Chem.MolToSmiles(mol) for mol in reference if mol is not None])
            nodes = max_atoms(train_loader, val_loader)
            no_bond_weight = measured_no_bond_weight(train_loader)

            for name in here:
                checkpoint = source / "runs" / f"seed{seed}_{name}" / "model.pt"
                model = load_weights(
                    source_module.build_model(
                        name, tokenizer, train_loader, train_table, val_table, nodes, seed,
                        device, no_bond_weight=no_bond_weight).to(device),
                    checkpoint, device)

                print(f"\n{dataset} seed{seed} {name}: {len(panel)} cold targets, {len(todo[name])} temperature(s) to do", flush=True)

                for temperature in todo[name]:
                    libraries = generate_libraries(
                        model, test_loader, panel,
                        n_generated=inherited["N_GENERATED"], temperature=temperature,
                        seed=seed, device=device, size_loader=train_loader)

                    directory = run_dir(dataset, EXPERIMENT, f"seed{seed}_{name}_{tag(temperature)}")
                    scores = score_libraries(
                        libraries, tokenizer, test_loader, metrics, oracles, panel,
                        reference, fingerprint_pca=fingerprint_pca,
                        success_threshold=inherited["SUCCESS_THRESHOLD"][dataset],
                        reference_activations=activations, device=device,
                        save_to=directory, save_scores=False, known_smiles=known)

                    record_run(directory, {"dataset": dataset, "seed": seed,
                                            "model": name, "temperature": temperature,
                                            "evaluation_split": "test",
                                            "n_targets": len(panel), **scores})
                    records, finished = collect_runs(output, ("seed", "model", "temperature"), valid=grid)
                    written += 1

                    got = records[-1]
                    print(f"  T={temperature:<5.2f} "
                            f"uniq {got.get('seq_uniqueness', float('nan')):.3f}  "
                            f"nov {got.get('seq_novelty', float('nan')):.3f}  "
                            f"scored {got.get('seq_MeanOracle_effective_delta', float('nan')):+.4f}  "
                            f"delta {got.get('seq_MeanOracle_delta_score', float('nan')):+.4f}",
                            flush=True)

                del model
                torch.cuda.empty_cache()

            del train_loader, val_loader, test_loader
            torch.cuda.empty_cache()

    frame = pd.DataFrame(records)
    print(f"\n{dataset}: {written} new scoring pass(es) in {(time.time() - started) / 60:.1f} min, {len(frame)} rows total")

    if analyse:
        write_analysis(frame, output, source, models, temperatures, seeds)

    return frame


def tag(temperature):
    return f"t{temperature:.2f}".replace(".", "p")


def write_analysis(frame, output, source=None, models=None, temperatures=None, seeds=None):
    """Curves, the per-temperature table, where each metric peaks, and the T = 1 control."""
    if frame.empty:
        return

    # the source experiment's directory
    source = Path(output).parent / SOURCE if source is None else source
    models = MODELS if models is None else models
    temperatures = TEMPERATURES if temperatures is None else temperatures
    seeds = SEEDS if seeds is None else seeds

    present = set(frame["model"])
    order = ([name for name in ARM_ORDER if name in present]
                + sorted(present - set(ARM_ORDER)))

    plot_metric_curves(frame, "temperature", "model", output / "figures", order=order, reference=REFERENCE_TEMPERATURE)

    columns = [c for c in frame.columns if c.startswith(("seq_", "graph_"))
                and not c.endswith(("_reference_mean_atoms", "_n_targets", "_n"))]
    summary = (frame.groupby(["model", "temperature"])[columns]
                .agg(["mean", "std", "count"])
                .stack(level=0, future_stack=True)
                .reset_index()
                .rename(columns={"level_2": "metric"}))
    summary.to_csv(output / "summary.csv", index=False)

    knee(frame, output, order)
    control(frame, source, output)

    done = frame.groupby(["model", "temperature"])["seed"].nunique()
    complete = int((done >= len(seeds)).sum())
    print(f"  analysis refreshed: {complete} of {len(models) * len(temperatures)} (model, temperature) cells have all {len(seeds)} seeds")


def knee(frame, output, order):
    """Where each metric peaks, per model, against its value at T = 1."""
    rows = []
    for model, block in frame.groupby("model"):
        for name in KNEE_METRICS:
            column = f"seq_{name}"
            if column not in block or block[column].isna().all():
                continue

            curve = block.groupby("temperature")[column].mean().dropna()
            if curve.empty:
                continue

            best = curve.idxmax()
            base = curve.get(REFERENCE_TEMPERATURE, float("nan"))
            rows.append({"model": model, "metric": name, "best_temperature": best,
                            "best": curve[best], "at_T1": base,
                            "gain": curve[best] - base})

    table = pd.DataFrame(rows)
    if table.empty:
        return

    table.to_csv(output / "knee.csv", index=False)
    print(f"\n{'=' * 78}\nwhere each metric peaks\n{'=' * 78}")
    for model in order:
        block = table[table.model == model]
        if not block.empty:
            print(f"\n{model}")
            print(block[["metric", "best_temperature", "best", "at_T1", "gain"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print(f"\n{'=' * 78}\nwhat temperature buys and costs, seq branch\n{'=' * 78}")
    for model in order:
        block = frame[frame.model == model]
        picked = [f"seq_{name}" for name in WATCH if f"seq_{name}" in block.columns]
        if not picked:
            continue
        curve = block.groupby("temperature")[picked].mean()
        curve.columns = [c.replace("seq_", "").replace("MeanOracle_", "") for c in curve.columns]
        print(f"\n{model}")
        print(curve.T.to_string(float_format=lambda v: f"{v:+.4f}"))


def control(frame, source, output):
    """T = 1 must reproduce the source run."""
    original = source / "results.csv"
    if not original.exists():
        return

    try:
        old = pd.read_csv(original)
    except (pd.errors.EmptyDataError, OSError):
        return

    here = frame[frame["temperature"] == REFERENCE_TEMPERATURE]
    checks = ["seq_uniqueness", "seq_novelty", "seq_MeanOracle_effective_delta", "seq_MeanOracle_delta_score"]

    rows = []
    unmatched = []
    for model, block in here.groupby("model"):
        was = old[old.model == model]
        for column in checks:
            if column not in block or column not in was or not was[column].notna().any():
                continue
            joined = block[["seed", column]].merge(was[["seed", column]], on="seed", suffixes=("_rerun", "_source"))
            unmatched.append(len(was) - len(joined))
            if joined.empty:
                rows.append({"model": model, "metric": column.replace("seq_", ""),
                                "source": was[column].mean(),
                                "T=1 rerun": block[column].mean()})
            else:
                rows.append({"model": model, "metric": column.replace("seq_", ""),
                                "source": joined[f"{column}_source"].mean(),
                                "T=1 rerun": joined[f"{column}_rerun"].mean()})

    if not rows:
        return

    table = pd.DataFrame(rows)
    table["difference"] = table["T=1 rerun"] - table["source"]
    table.to_csv(output / "control.csv", index=False)

    worst = table["difference"].abs().max()
    print(f"\n{'=' * 78}\ncontrol: T = 1 against {source.name}\n{'=' * 78}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\n  largest discrepancy {worst:.4f}. {'the harness reproduces the run' if worst < CONTROL_TOLERANCE else 'INVESTIGATE, this should be ~0'}")


def main():
    temperatures = sorted({round(float(t), 2) for t in TEMPERATURES})
    if REFERENCE_TEMPERATURE not in temperatures:
        temperatures.append(REFERENCE_TEMPERATURE)
        temperatures.sort()

    device = setup_compute()
    frames = []

    for dataset in DATASETS:
        print(chr(10) + "=" * 70 + chr(10) + f"=== {dataset}" + chr(10) + "=" * 70, flush=True)
        try:
            frames.append(run_dataset(dataset, device, list(MODELS), temperatures, list(SEEDS)))
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
