import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models.deepdtagen import DeepDTAGen
from models.prot2drug import Prot2Drug
from models.sel_vae_gen import build_selvaegen
from models.zerogen import ZeroGEN
from utils import save_figure
from utils.data_tools import (create_dataloaders, load_fingerprint_pca, load_tokenizer,
                            reset_loader_epoch, shared_drug_ids)
from utils.experiment_tools import (SUCCESS_THRESHOLD, MeanOracle, cold_targets,
                                    collect_runs, complete_blocks, describe_arms,
                                    env_seeds, experiment_dir, finish_run,
                                    generate_libraries, load_oracle, max_atoms,
                                    measured_no_bond_weight, plot_metric_boxes,
                                    record_run, reference_molecules,
                                    run_dir, run_training_concurrently, save_config,
                                    score_libraries, setup_compute, token_swap,
                                    training_smiles)
from utils.metrics import Metrics
from utils.selectivity import build_objective

DATASETS = ["kiba", "bindingdb", "papyrus", "davis"]

SEEDS = env_seeds("E6_SEEDS", range(5))
MODELS = ["SelVAEGen", "DeepDTAGen", "Prot2Drug", "ZeroGEN"]

CONCURRENT_MODELS = 4

EPOCHS = 150
PRETRAIN_EPOCHS = 100
PATIENCE = 30
BATCH_SIZE = 256
GROUPS_PER_BATCH = 16
GRAD_CLIP = 50.0
PANEL_SIZE = 3
FREEZE_PRIOR_EPOCHS = 10
FREEZE_PANEL = False

N_GENERATED = 256
N_REFERENCE = 2000
N_PANEL_TARGETS = None
N_EVAL_TARGETS = None
GENERATION_QUANTILE = 0.9

TRAINING = {
    # the paper's parameters
    "DeepDTAGen": dict(lr=2e-4, weight_decay=0.0, scheduler=None),
    "Prot2Drug": dict(lr=1e-3, weight_decay=0.0, scheduler=None),
    "ZeroGEN": dict(lr=1e-4, weight_decay=0.05, scheduler=(10, 0.5)),
}

REPORTING_ORACLES = ["GraphDTAOracle", "GSDTAOracle", "BaselineOracle"]
EXPERIMENT = "e6_compare"

SELVAEGEN_USE_SELFIES = True
SELVAEGEN_PARAMS_SOURCE = "e2 seed 0, chosen by hand; e5 not yet transcribed"
SELVAEGEN_BRANCH_FLAGS = dict(gnn_encoder=True, seq_encoder=True,
                                gnn_decoder=False, seq_decoder=True)
# update after e5
SELVAEGEN_PARAMS = dict(
    learning_rate=1e-4,
    weight_decay=1e-5,
    fusion_dim=512,
    panel_size=PANEL_SIZE,
    w_seq=4.0,
    w_graph=1.0,
    w_kl_mean=0.02,
    w_kl_var=1.0,
    w_affinity=1.0,
    w_selectivity=1.0,
    w_distributional=1.0,
    affinity_rank_margin=10.0,
    selectivity_rank_margin=10.0,
    direction_weight=0.5,
    distribution_mean_margin=10.0,
    distribution_effect_margin=2.0,
)

OPTIMIZER_KEYS = ("learning_rate", "weight_decay")
MODEL_KEYS = ("fusion_dim",)
OBJECTIVE_KEYS = ("panel_size",)


def create_model_loadout(dataset, tokenizer, shared, seed, device):
    return create_dataloaders(
        dataset, tokenizer=tokenizer, batch_size=BATCH_SIZE, train_frac=0.8,
        val_frac=0.1, test_frac=0.1, random_state=seed, device=device,
        groups_per_batch=GROUPS_PER_BATCH, restrict_drug_ids=shared)


def _assert_split_identity(left, right):
    columns = ["drug_id", "protein_id", "affinity"]
    for label, left_table, right_table in zip(("train", "validation", "test"), left, right):
        if not left_table[columns].reset_index(drop=True).equals(
                right_table[columns].reset_index(drop=True)):
            raise RuntimeError(f"SMILES and SELFIES produced different {label} splits")


def split_params(params):
    optimizer = {key: params[key] for key in OPTIMIZER_KEYS if key in params}
    model = {key: params[key] for key in MODEL_KEYS if key in params}
    objective = {key: params[key] for key in OBJECTIVE_KEYS if key in params}
    loss = {key: value for key, value in params.items() if key not in OPTIMIZER_KEYS + MODEL_KEYS + OBJECTIVE_KEYS}
    return optimizer, model, objective, loss


def build_model(name, tokenizer, loader, train_table, val_table, nodes, seed, device, no_bond_weight=1.0):
    protein_dim = loader.protein_embs.shape[1]
    smiles_len = loader.drug_token_table.shape[1]

    if name == "SelVAEGen":
        _, model_params, objective_params, loss_params = split_params(SELVAEGEN_PARAMS)
        criterion = build_objective(
            train_table, loader.protein_embs, len(loader.drug_token_table),
            len(loader.protein_table), pad_id=tokenizer.pad_id, seed=seed, device=device,
            freeze_panel=FREEZE_PANEL, val_table=val_table,
            no_bond_weight=no_bond_weight, **objective_params, **loss_params)
        return build_selvaegen(
            tokenizer, protein_dim, nodes, smiles_len, criterion=criterion,
            fingerprint_dim=loader.fingerprint_table.shape[1],
            **SELVAEGEN_BRANCH_FLAGS, **model_params)

    if name == "DeepDTAGen":
        return DeepDTAGen(
            tokenizer, max_length=smiles_len,
            generation_affinity=float(train_table["affinity"].quantile(GENERATION_QUANTILE)))
    if name == "Prot2Drug":
        return Prot2Drug(tokenizer, protein_dim, max_length=smiles_len)
    return ZeroGEN(tokenizer, protein_dim, max_length=smiles_len,
                    affinity_min=float(train_table["affinity"].min()),
                    affinity_max=float(train_table["affinity"].max()))


def pretrain_zerogen(model, train_loader, settings):
    pretrainer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    try:
        model.train()
        for epoch in range(1, PRETRAIN_EPOCHS + 1):
            total, batches = 0.0, 0
            for batch in train_loader:
                loss, values = model.pretrain_loss(batch)
                pretrainer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
                if not np.isfinite(norm):
                    pretrainer.zero_grad(set_to_none=True)
                    continue
                pretrainer.step()
                total += values["loss"]
                batches += 1

            if not batches:
                raise RuntimeError(f"ZeroGEN pretraining epoch {epoch} had no finite batches")
            print(f"  mlm epoch {epoch:3d} | loss {total / batches:.3f}")
        model.sync_decoder_self_attention()
    finally:
        del pretrainer


def run_dataset(dataset, device):
    """The rows from every model over every seed for one dataset."""
    seeds = list(SEEDS)
    output = experiment_dir(dataset, EXPERIMENT)
    records, finished = collect_runs(output, ("seed", "model"), valid={"seed": seeds})
    save_config(output, sys.modules[__name__], dataset=dataset)

    pending = {(seed, name) for seed in seeds for name in MODELS if not finished(seed=seed, model=name)}
    if not pending:
        # nothing left to train
        frame = pd.DataFrame(records)
        write_analysis(frame, output)

        return frame

    tokenizers = {
        "selvaegen": load_tokenizer(dataset, use_selfies=SELVAEGEN_USE_SELFIES),
        "baseline": load_tokenizer(dataset, use_selfies=False),
    }
    shared = shared_drug_ids(dataset, tokenizers.values())
    metrics = Metrics()
    oracles = {name: load_oracle(dataset, name) for name in REPORTING_ORACLES}
    oracles["MeanOracle"] = MeanOracle(list(oracles.values())).to(device).eval()
    fingerprint_pca = load_fingerprint_pca(dataset)

    for seed in SEEDS:
        if not any((seed, name) in pending for name in MODELS):
            continue

        loadouts = {kind: create_model_loadout(dataset, tokenizer, shared, seed, device) for kind, tokenizer in tokenizers.items()}
        _assert_split_identity(loadouts["selvaegen"][1], loadouts["baseline"][1])

        sel_train_loader, _, sel_test_loader = loadouts["selvaegen"][0]
        panel_ids = cold_targets(sel_test_loader)
        if N_PANEL_TARGETS is not None:
            panel_ids = panel_ids[:N_PANEL_TARGETS]

        reference = reference_molecules(sel_train_loader, N_REFERENCE)
        known = training_smiles(sel_train_loader)
        activations = metrics.chemnet_activations([Chem.MolToSmiles(mol) for mol in reference if mol is not None])

        due = [name for name in MODELS if (seed, name) in pending]
        arms, built = [], {}

        for group in [due[i:i + CONCURRENT_MODELS] for i in range(0, len(due), CONCURRENT_MODELS)]:
            arms = []
            for name in group:
                torch.manual_seed(seed)
                kind = "selvaegen" if name == "SelVAEGen" else "baseline"
                (train_loader, val_loader, test_loader), (train_table, val_table, _) = loadouts[kind]
                tokenizer = tokenizers[kind]

                model = build_model(
                    name, tokenizer, train_loader, train_table, val_table,
                    max_atoms(train_loader, val_loader), seed, device,
                    no_bond_weight=measured_no_bond_weight(train_loader)).to(device)

                if name == "SelVAEGen":
                    optimizer_params, _, _, _ = split_params(SELVAEGEN_PARAMS)
                    settings = dict(lr=optimizer_params["learning_rate"],
                                    weight_decay=optimizer_params["weight_decay"],
                                    scheduler=None)
                else:
                    settings = dict(TRAINING[name])

                optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
                scheduler = None
                if settings["scheduler"]:
                    step, gamma = settings["scheduler"]
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step, gamma)

                print(f"\n{dataset} | seed {seed} | {name} | {sum(p.numel() for p in model.parameters()):,} parameters")

                if name == "ZeroGEN":
                    pretrain_zerogen(model, train_loader, settings)
                    torch.cuda.empty_cache()

                def start(epoch, name=name, model=model):
                    if name == "SelVAEGen":
                        model.set_protein_prior_trainable(epoch > FREEZE_PRIOR_EPOCHS)
                        model.criterion.set_epoch(epoch)

                def finish(epoch, name=name, model=model):
                    if name == "ZeroGEN":
                        model.on_epoch_end()

                arms.append({"label": name, "model": model, "optimizer": optimizer,
                                "scheduler": scheduler, "on_epoch_start": start,
                                "on_epoch_end": finish,
                                "prepare": token_swap(train_loader.drug_token_table),
                                "diagnostics": (model.criterion.geometry if name == "SelVAEGen" else None)})
                built[name] = dict(kind=kind, tokenizer=tokenizer, model=model,
                                    train_loader=train_loader, test_loader=test_loader,
                                    optimizer=optimizer, scheduler=scheduler)

            # one shared loader
            lead = built[group[0]]["train_loader"]
            lead_val = loadouts[built[group[0]]["kind"]][0][1]
            reset_loader_epoch(lead)

            started = time.perf_counter()
            trained = run_training_concurrently(
                arms, lead, lead_val, EPOCHS, grad_clip=GRAD_CLIP, patience=PATIENCE,
                label=f"{dataset} seed{seed} ")

            written = {}
            for name in group:
                directory = run_dir(dataset, EXPERIMENT, f"seed{seed}_{name}")
                written[name] = (directory, finish_run(
                    trained[name][0], built[name]["model"],
                    time.perf_counter() - started, directory,
                    grad_clip=GRAD_CLIP, state=trained[name][1]))

            failed = []
            for name in group:
                made = built[name]
                kind, tokenizer = made["kind"], made["tokenizer"]
                model, train_loader = made["model"], made["train_loader"]
                test_loader = made["test_loader"]
                history, _, best_epoch, best_loss = trained[name]
                directory, summary = written[name]

                try:
                    libraries = generate_libraries(
                        model, test_loader,
                        panel_ids if N_EVAL_TARGETS is None else panel_ids[:N_EVAL_TARGETS],
                        n_generated=N_GENERATED, seed=seed, device=device,
                        size_loader=train_loader)
                    scores = score_libraries(
                        libraries, tokenizer, test_loader, metrics, oracles, panel_ids,
                        reference, fingerprint_pca=fingerprint_pca,
                        success_threshold=SUCCESS_THRESHOLD[dataset],
                        reference_activations=activations, device=device,
                        save_to=directory, known_smiles=known)
                except Exception as error:
                    failed.append(name)
                    print(f"  seed{seed} {name}: scoring failed, weights kept at "
                            f"{directory / 'model.pt'} -- {type(error).__name__}: {error}", flush=True)
                    continue

                records.append({
                    "dataset": dataset,
                    "seed": seed,
                    "split_seed": seed,
                    "evaluation_split": "test",
                    "model": name,
                    "model_tokens": ("selfies" if SELVAEGEN_USE_SELFIES else "smiles") if kind == "selvaegen" else "smiles",
                    "hyperparams_source": (SELVAEGEN_PARAMS_SOURCE if name == "SelVAEGen" else "published defaults"),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_loss,
                    **summary,
                    **history[best_epoch - 1],
                    **scores,
                })
                record_run(directory, records[-1])
                records, finished = collect_runs(output, ("seed", "model"), valid={"seed": SEEDS})

            if failed:
                print(f"  seed{seed}: {len(failed)} of {len(group)} models unscored "
                        f"({', '.join(failed)}); the rest were recorded and only these "
                        f"will be retrained", flush=True)

            for made in built.values():
                made["model"] = made["optimizer"] = made["scheduler"] = None
            arms.clear()
            torch.cuda.empty_cache()

        write_analysis(pd.DataFrame(records), output)

    return pd.DataFrame(records)


def main():
    """Each dataset in turn, resuming wherever the last run stopped."""
    device = setup_compute()
    frames = []

    for dataset in DATASETS:
        print(chr(10) + "=" * 70 + chr(10) + f"=== {dataset}" + chr(10) + "=" * 70, flush=True)
        try:
            frames.append(run_dataset(dataset, device))
        except FileNotFoundError as missing:
            print(f"  {dataset}: skipped, {missing}", flush=True)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_analysis(frame, output):
    """Summary and the per-branch figures, for whatever seeds are complete."""
    if frame.empty:
        return

    # summary and figures describe the same seeds
    frame = complete_blocks(frame, "model")
    describe_arms(frame, "model", output / "summary.csv")

    plot_metric_boxes(frame, "model", output / "figures", order=MODELS)
    plot_distributional_separation(frame, output, gaps=distributional_gap(frame, output))

    done = frame.groupby("seed")["model"].nunique()
    complete = int((done >= len(MODELS)).sum())
    print(f"  analysis refreshed: {len(frame)} runs, {complete} complete seeds of {len(SEEDS)}")


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"
ON_COLOUR = "#2a78d6"
OFF_COLOUR = "#eb6834"
MODEL_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def style_axes(axes):
    axes.set_facecolor(SURFACE)
    axes.set_axisbelow(True)
    axes.grid(True, color=GRIDLINE, linewidth=0.6)
    axes.tick_params(colors=INK_MUTED, labelsize=8.5, length=3, width=0.8)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS_RULE)
        axes.spines[side].set_linewidth(0.8)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_color(INK_SECONDARY)


def seed_affinities(output, model, seed):
    path = Path(output) / "runs" / f"seed{int(seed)}_{model}" / "affinities.npz"

    if not path.exists():
        return None, None

    matrices = np.load(path, allow_pickle=False)
    on_target, off_target = [], []

    for target in range(len(matrices["panel_ids"])):
        key = f"seq|MeanOracle|{target}"
        if key not in matrices.files:
            continue
        affinities = matrices[key]
        on_target.append(affinities[:, target])
        off_target.append(np.delete(affinities, target, axis=1).reshape(-1))

    if not on_target:
        return None, None

    return np.concatenate(on_target), np.concatenate(off_target)


def pooled_affinities(output, block, model):
    runs = [seed_affinities(output, model, run["seed"]) for _, run in block.iterrows()]
    kept = [(on_target, off_target) for on_target, off_target in runs if on_target is not None]

    if not kept:
        return None, None

    return (np.concatenate([on_target for on_target, _ in kept]),
            np.concatenate([off_target for _, off_target in kept]))


def survival(grid, curve):
    """Mass of `curve` above every point of `grid`, by trapezoid accumulated from the right."""
    steps = np.diff(grid) * 0.5 * (curve[:-1] + curve[1:])

    return np.concatenate([np.cumsum(steps[::-1])[::-1], [0.0]])


def distributional_gap(frame, output, max_kde_points=50000, cuts=2000):
    """Per seed, the threshold-to-infinity integral of on-target minus off-target density."""
    from scipy.stats import gaussian_kde

    dataset = str(frame["dataset"].iloc[0])
    threshold = SUCCESS_THRESHOLD[dataset]
    rng = np.random.default_rng(0)
    runs, floor, ceiling = [], [], []

    def estimator(values):
        sample = (rng.choice(values, max_kde_points, replace=False) if len(values) > max_kde_points else values)
        return gaussian_kde(sample)

    for model in MODELS:
        for seed in sorted(frame.loc[frame["model"] == model, "seed"].unique()):
            on_target, off_target = seed_affinities(output, model, seed)
            if on_target is None:
                continue
            runs.append({"dataset": dataset, "model": model, "seed": int(seed), "threshold": threshold,
                            "on_kde": estimator(on_target), "off_kde": estimator(off_target)})
            floor.append(float(min(on_target.min(), off_target.min())))
            ceiling.append(float(max(on_target.max(), off_target.max())))
            del on_target, off_target

    if not runs:
        return pd.DataFrame()

    margin = 0.05 * (max(ceiling) - min(floor))
    cut = np.linspace(min(floor) - margin, max(ceiling) + margin, cuts)
    index = int(np.searchsorted(cut, threshold))

    for run in runs:
        on_above = survival(cut, run.pop("on_kde")(cut))
        off_above = survival(cut, run.pop("off_kde")(cut))
        run["on_target_tail_mass"] = float(on_above[index])
        run["off_target_tail_mass"] = float(off_above[index])
        run["tail_mass_gap"] = float(on_above[index] - off_above[index])

    by_seed = pd.DataFrame(runs)
    summary = by_seed.groupby("model", sort=False).agg(
        seeds=("seed", "count"),
        on_target_tail_mass=("on_target_tail_mass", "mean"),
        on_target_tail_mass_std=("on_target_tail_mass", "std"),
        off_target_tail_mass=("off_target_tail_mass", "mean"),
        off_target_tail_mass_std=("off_target_tail_mass", "std"),
        tail_mass_gap=("tail_mass_gap", "mean"),
        tail_mass_gap_std=("tail_mass_gap", "std")).reset_index()
    summary.insert(0, "dataset", dataset)
    summary.insert(2, "threshold", threshold)

    summary.rename(columns={"tail_mass_gap": "mean_gap", "tail_mass_gap_std": "std"})[
        ["model", "mean_gap", "std"]].to_csv(Path(output) / "distributional_gap.csv", index=False)

    print(f"  distributional gap above {threshold:g} ({dataset}), mean over seeds:")
    for row in summary.itertuples():
        print(f"    {row.model:12s} on {row.on_target_tail_mass:.4f}  off {row.off_target_tail_mass:.4f}"
                f"  gap {row.tail_mass_gap:+.4f} +- {row.tail_mass_gap_std:.4f}  ({row.seeds} seeds)")

    return summary


def estimate_panels(frame, output, threshold, max_kde_points, rng):
    """One density estimator per model, with the window and counts needed to draw them together."""
    from scipy.stats import gaussian_kde

    metric = "seq_MeanOracle_distributional_delta_score"
    panels = []

    def estimator(values):
        sample = (rng.choice(values, max_kde_points, replace=False) if len(values) > max_kde_points else values)
        return gaussian_kde(sample)

    for model in MODELS:
        block = frame[frame["model"] == model]
        on_target, off_target = pooled_affinities(output, block, model)

        if on_target is None:
            continue

        pooled = np.concatenate([on_target, off_target])
        low, high = np.quantile(pooled, [0.005, 0.995])
        values = block[metric].dropna().to_numpy(dtype=float)
        panels.append({"model": model, "on_kde": estimator(on_target), "off_kde": estimator(off_target),
                        "low": float(low), "high": float(high),
                        "floor": float(pooled.min()), "ceiling": float(pooled.max()),
                        "on_target_n": len(on_target), "off_target_n": len(off_target),
                        "seeds": len(values),
                        "delta_mean": float(values.mean()) if len(values) else float("nan"),
                        "delta_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan")})
        # the raw panels run to millions of rows, so only the capped KDE sample stays alive
        del on_target, off_target, pooled

    return panels


def plot_distributional_separation(frame, output, gaps=None, max_kde_points=50000, cuts=2000):
    """Pooled on- and off-target MeanOracle densities on shared axes, one panel per model."""
    from matplotlib import pyplot as plt

    spread = ({} if gaps is None or gaps.empty
                else gaps.set_index("model")[["on_target_tail_mass", "off_target_tail_mass",
                                                "tail_mass_gap", "tail_mass_gap_std"]].to_dict("index"))
    dataset = str(frame["dataset"].iloc[0])
    threshold = SUCCESS_THRESHOLD[dataset]
    directory = Path(output) / "figures" / "distributional_separation"
    directory.mkdir(parents=True, exist_ok=True)
    panels = estimate_panels(frame, output, threshold, max_kde_points, np.random.default_rng(0))

    if not panels:
        return

    low = min(panel["low"] for panel in panels)
    high = max(panel["high"] for panel in panels)
    pad = 0.08 * ((high - low) or 1.0)
    grid = np.linspace(low - pad, high + pad, 400)
    # the integrals get their own wider grid, past the tails the window clips
    margin = 0.05 * (max(p["ceiling"] for p in panels) - min(p["floor"] for p in panels))
    cut = np.linspace(min(p["floor"] for p in panels) - margin,
                        max(p["ceiling"] for p in panels) + margin, cuts)
    at_threshold = int(np.searchsorted(cut, threshold))

    for panel in panels:
        panel["grid"], panel["cut"] = grid, cut
        panel["on_density"], panel["off_density"] = panel["on_kde"](grid), panel["off_kde"](grid)
        panel["on_above"] = survival(cut, panel["on_kde"](cut))
        panel["off_above"] = survival(cut, panel["off_kde"](cut))
        panel["gap"] = panel["on_above"] - panel["off_above"]
        peak = int(panel["gap"].argmax())
        panel["scalars"] = {"distributional_delta_mean": panel["delta_mean"],
                            "distributional_delta_std": panel["delta_std"],
                            "seeds": panel["seeds"], "on_target_n": panel["on_target_n"],
                            "off_target_n": panel["off_target_n"], "threshold": threshold,
                            "on_target_mass": float(panel["on_above"][0]),
                            "off_target_mass": float(panel["off_above"][0]),
                            "on_target_tail_mass": float(panel["on_above"][at_threshold]),
                            "off_target_tail_mass": float(panel["off_above"][at_threshold]),
                            "tail_mass_gap": float(panel["gap"][at_threshold]),
                            "best_cut": float(cut[peak]),
                            "best_tail_mass_gap": float(panel["gap"][peak])}

    def draw(axes, panel, title=True):
        density_on, density_off = panel["on_density"], panel["off_density"]
        tail = grid >= threshold
        axes.axvline(threshold, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.0,
                        label=f"Threshold {threshold:g}", zorder=1)
        axes.fill_between(grid, density_on, density_off, where=tail & (density_on >= density_off),
                            color=ON_COLOUR, alpha=0.22, linewidth=0, zorder=2,
                            label="On-target excess above threshold")
        axes.fill_between(grid, density_on, density_off, where=tail & (density_off > density_on),
                            color=OFF_COLOUR, alpha=0.22, linewidth=0, zorder=2,
                            label="Off-target excess above threshold")
        axes.plot(grid, density_off, color=OFF_COLOUR, linewidth=2.0, label="Off-target", zorder=3)
        axes.plot(grid, density_on, color=ON_COLOUR, linewidth=2.0, label="On-target", zorder=4)

        scalars = panel["scalars"]
        axes.text(0.035, 0.955, panel["model"] if title else "", transform=axes.transAxes,
                    va="top", ha="left", fontsize=11, color=INK)
        quoted = spread.get(panel["model"], scalars)
        over_seeds = spread.get(panel["model"])
        axes.text(0.975, 0.955, f"on {quoted['on_target_tail_mass']:.4f}    "
                                f"off {quoted['off_target_tail_mass']:.4f}",
                    transform=axes.transAxes, va="top", ha="right", fontsize=9, color=INK_SECONDARY)
        label = (f"gap {scalars['tail_mass_gap']:+.4f}" if over_seeds is None else
                    f"gap {over_seeds['tail_mass_gap']:+.4f} $\\pm$ {over_seeds['tail_mass_gap_std']:.4f}")
        axes.text(0.975, 0.868, label, transform=axes.transAxes,
                    va="top", ha="right", fontsize=11.5, color=INK)
        axes.set_ylim(0, 1.28 * max(density_on.max(), density_off.max()))
        axes.set_xlim(grid[0], grid[-1])
        style_axes(axes)

    def bottom_legend(figure, axes, ncol=5):
        handles, labels = axes.get_legend_handles_labels()
        order = [labels.index(name) for name in ("On-target", "Off-target",
                                                    "On-target excess above threshold",
                                                    "Off-target excess above threshold",
                                                    f"Threshold {threshold:g}")]
        figure.legend([handles[index] for index in order], [labels[index] for index in order],
                        loc="lower center", ncol=ncol, frameon=False, fontsize=9,
                        labelcolor=INK_SECONDARY)

    for panel in panels:
        figure, axes = plt.subplots(figsize=(6.8, 4.4))
        figure.patch.set_facecolor(SURFACE)
        draw(axes, panel)
        axes.set_xlabel("Predicted affinity", color=INK_SECONDARY, fontsize=9.5)
        axes.set_ylabel("Density", color=INK_SECONDARY, fontsize=9.5)
        bottom_legend(figure, axes, ncol=3)
        figure.tight_layout(rect=(0, 0.12, 1, 1))

        path = directory / f"{panel['model']}.pdf"
        save_figure(figure, path, dpi=200, close=False)
        panel_table(panel, dataset).to_csv(path.with_suffix(".csv"), index=False)
        plt.close(figure)

    figure, cells = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.9), sharex=True)
    figure.patch.set_facecolor(SURFACE)
    cells = np.atleast_1d(cells)
    for axes, panel in zip(cells, panels):
        draw(axes, panel)
        axes.set_xlabel("Predicted affinity", color=INK_SECONDARY, fontsize=9.5)
    cells[0].set_ylabel("Density", color=INK_SECONDARY, fontsize=9.5)

    bottom_legend(figure, cells[0])
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    path = directory / "models_side_by_side.pdf"
    pd.concat([panel_table(panel, dataset) for panel in panels],
                ignore_index=True).to_csv(path.with_suffix(".csv"), index=False)
    save_figure(figure, path, dpi=200)

    plot_tail_mass_sweep(panels, dataset, threshold, directory)


def plot_tail_mass_sweep(panels, dataset, threshold, directory):
    from matplotlib import pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.1), sharex=True)
    figure.patch.set_facecolor(SURFACE)
    tables = []

    for index, panel in enumerate(panels):
        colour = MODEL_COLOURS[index % len(MODEL_COLOURS)]
        left.plot(panel["cut"], panel["on_above"], color=colour, linewidth=2.0, label=panel["model"])
        right.plot(panel["cut"], panel["gap"], color=colour, linewidth=2.0, label=panel["model"])
        tables.append(pd.DataFrame({"dataset": dataset, "model": panel["model"], "cut": panel["cut"],
                                    "on_target_tail_mass": panel["on_above"],
                                    "off_target_tail_mass": panel["off_above"],
                                    "tail_mass_gap": panel["gap"], "threshold": threshold}))

    leader = max(panels, key=lambda panel: panel["scalars"]["best_tail_mass_gap"])
    best_cut = leader["scalars"]["best_cut"]
    best_gap = leader["scalars"]["best_tail_mass_gap"]
    right.plot([best_cut], [best_gap], marker="o", markersize=6, zorder=5,
                color=MODEL_COLOURS[panels.index(leader) % len(MODEL_COLOURS)],
                markeredgecolor=SURFACE, markeredgewidth=2)
    right.set_title(f"widest gap   {leader['model']}  {best_gap:+.3f} at cut {best_cut:.2f}",
                    loc="right", fontsize=9, color=INK, pad=6)

    right.axhline(0.0, color=AXIS_RULE, linewidth=0.8, zorder=1)
    left.set_ylabel("On-target mass above the cut", color=INK_SECONDARY, fontsize=9.5)
    right.set_ylabel("On-target minus off-target mass above the cut", color=INK_SECONDARY, fontsize=9.5)

    for axes in (left, right):
        axes.axvline(threshold, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
        axes.annotate(f"threshold {threshold:g}", xy=(threshold, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(4, -10), textcoords="offset points", fontsize=8.5, color=INK_MUTED)
        axes.set_xlabel("Cut on predicted affinity", color=INK_SECONDARY, fontsize=9.5)
        style_axes(axes)

    left.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK_SECONDARY)
    figure.suptitle(f"{dataset} — separation against the choice of cut", fontsize=11.5,
                    color=INK, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = directory / "tail_mass_sweep.pdf"
    pd.concat(tables, ignore_index=True).to_csv(path.with_suffix(".csv"), index=False)
    save_figure(figure, path, dpi=200)


def panel_table(panel, dataset):
    """The panel's two densities on their shared grid, with its scalars repeated per row."""
    return pd.DataFrame({"affinity": panel["grid"], "on_target_density": panel["on_density"],
                            "off_target_density": panel["off_density"], "dataset": dataset,
                            "model": panel["model"], **panel["scalars"]})


if __name__ == "__main__":
    frame = main()

    if frame.empty:
        raise SystemExit("no dataset produced any runs")

    print("\n=== does the ranking survive the choice of oracle? ===")
    for metric in ("delta_score", "directional_consistency", "success_rate"):
        picked = [column for column in (f"seq_{name}_{metric}" for name in REPORTING_ORACLES + ["MeanOracle"]) if column in frame]
        print(f"\n{metric}:")
        print(frame.groupby(["dataset", "model"])[picked].mean().to_string(float_format=lambda value: f"{value:.4f}"))
