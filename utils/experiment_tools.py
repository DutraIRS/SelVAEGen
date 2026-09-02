import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import RESULTS
from .chemical_tools import to_mols
from .data_tools import build_generated_loader
from .experiment_plots import (
    describe_arms,
    plot_decision,
    plot_grad_norm,
    plot_losses,
    plot_metric_boxes,
    plot_metric_curves,
)
from .experiment_results import (
    BRANCH_AGGREGATION,
    DECISION_METRIC,
    LIBRARY_METRICS,
    ORACLE_METRICS,
    ORACLE_NAMES,
    PRIMARY_METRICS,
    SUCCESS_THRESHOLD,
    branch_decision,
    collect_runs,
    complete_blocks,
    decision_column,
    env_seeds,
    experiment_dir,
    grid_complete,
    metric_family,
    provenance,
    record_run,
    run_dir,
    save_config,
)

__all__ = (
    "BRANCH_AGGREGATION",
    "DECISION_METRIC",
    "LIBRARY_METRICS",
    "ORACLE_METRICS",
    "ORACLE_NAMES",
    "PRIMARY_METRICS",
    "SUCCESS_THRESHOLD",
    "MeanOracle",
    "branch_decision",
    "cold_targets",
    "collect_runs",
    "complete_blocks",
    "decision_column",
    "describe_arms",
    "env_seeds",
    "evaluate_generator",
    "experiment_dir",
    "finish_run",
    "generate_libraries",
    "grid_complete",
    "load_oracle",
    "load_weights",
    "matched_init",
    "max_atoms",
    "measured_no_bond_weight",
    "metric_family",
    "panel_loader",
    "plot_decision",
    "plot_grad_norm",
    "plot_losses",
    "plot_metric_boxes",
    "plot_metric_curves",
    "provenance",
    "record_run",
    "reference_molecules",
    "run_dir",
    "run_training",
    "run_training_concurrently",
    "save_config",
    "save_libraries",
    "score_libraries",
    "score_with",
    "setup_compute",
    "token_swap",
    "training_smiles",
)


def matched_init(models, reference=None):
    """Give every arm the same weights wherever their modules line up."""
    models = list(models)
    if not models:
        return models

    if reference is None:
        reference = max(models, key=lambda m: sum(p.numel() for p in m.parameters()))

    source = reference if isinstance(reference, dict) else reference.state_dict()

    for model in models:
        target = model.state_dict()
        shared = {name: source[name] for name, tensor in target.items() if name in source and source[name].shape == tensor.shape}
        model.load_state_dict(shared, strict=False)

    return models


class MeanOracle(torch.nn.Module):
    """Several oracles averaged, so nothing downstream tunes against one model's quirks."""
    def __init__(self, oracles):
        super().__init__()
        self.oracles = torch.nn.ModuleList(oracles)

    def forward(self, batch):
        return torch.stack([oracle(batch) for oracle in self.oracles], dim=0).mean(0)


def load_oracle(dataset, name=ORACLE_NAMES, device=None):
    """The pretrained affinity oracle(s), as saved whole by e1_train_oracles."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load(single):
        return torch.load(RESULTS / dataset / "oracles" / f"{single}.pt", map_location=device, weights_only=False).to(device).eval()

    if isinstance(name, str):
        return load(name)

    return MeanOracle([load(single) for single in name]).to(device).eval()


def cold_targets(loader):
    """Targets behind a validation or test loader, ordered by available measurements."""
    counts = Counter(int(point.protein_id) for point in loader.dataset)

    return [protein for protein, _ in counts.most_common()]

def max_atoms(*loaders):
    """Widest molecule behind any loader; the graph decoder needs that many node slots."""
    return max(int(point.num_nodes) for loader in loaders for point in loader.dataset)

def measured_no_bond_weight(loader, softening=0.5):
    """The edge class weight: data-driven, but pulled back from exact class balancing."""
    bonded = pairs = 0
    for point in loader.dataset:
        nodes = int(point.num_nodes)
        bonded += int(point.edge_index.shape[1])
        pairs += nodes * (nodes - 1)

    if bonded == 0:
        return 1.0

    return (bonded / max(pairs - bonded, 1)) ** softening

def setup_compute(tf32=True):
    """Pick the device and let Ampere use its tensor cores for fp32 matmuls."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return device

def training_smiles(loader):
    from rdkit import Chem

    known = set()
    for smiles in {point.smiles for point in loader.dataset}:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            known.add(Chem.MolToSmiles(mol, canonical=True))

    return known

def reference_molecules(loader, count=2000, seed=0):
    """Training molecules standing in as the novelty and FCD reference."""
    smiles = sorted({point.smiles for point in loader.dataset})

    if len(smiles) > count:
        rows = np.random.default_rng(seed).choice(len(smiles), count, replace=False)
        smiles = [smiles[row] for row in sorted(rows)]

    return to_mols(smiles)


def finish_run(history, model, seconds, directory, grad_clip=None, state=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(history)
    if grad_clip is not None:
        frame["grad_clip"] = grad_clip
    frame.to_csv(directory / "history.csv", index=False)

    if state is not None:
        torch.save({"state_dict": state, "model": type(model).__name__,
                    "epochs_run": len(history), **provenance()}, directory / "model.pt")

    plot_losses(frame, directory / "loss.pdf")
    plot_grad_norm(frame, directory / "grad_norm.pdf")

    peak = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0

    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "train_seconds": seconds,
            "epochs_run": len(history),
            "seconds_per_epoch": seconds / max(len(history), 1),
            "peak_memory_gb": peak,
            "grad_norm_final": frame["grad_norm"].iloc[-1],
            "grad_clipped_final": frame["grad_clipped"].iloc[-1],
            **provenance()}

LOSS_TERMS = ("kl_mean", "kl_var", "recon_seq", "recon_graph", "affinity", "selectivity", "distributional")

def _present(row, name):
    value = row.get(name)

    return value is not None and pd.notna(value)

def _pair(row, name, digits=3):
    """"<name> train|val" for one loss term, or None when the model has no such term."""
    train = row.get(name)

    if train is None or pd.isna(train):               # a branch this model does not have
        return None

    text = f"{train:.{digits}f}"
    val = row.get(f"val_{name}")

    if val is not None and pd.notna(val):
        text += f"|{val:.{digits}f}"

    return f"{name} {text}"


def format_epoch(row, epochs, best_score, best_epoch, waited, patience, label=""):
    """One epoch as three grouped lines instead of thirty fields on one."""
    stamp = time.strftime("%H:%M:%S")
    waiting = f"  patience {waited}/{patience}" if patience is not None else ""

    terms = [text for text in (_pair(row, name) for name in LOSS_TERMS) if text]

    diagnostics = []
    if _present(row, "dist_gap_mean"):
        spread = row["dist_gap_std"] if _present(row, "dist_gap_std") else float("nan")
        diagnostics.append(f"gap {row['dist_gap_mean']:.2f}+-{spread:.2f}")
    for name, text in (("dist_effect", "d"), ("centre_spacing", "centres"), ("centre_spacing_ratio", "ratio")):
        if _present(row, name):
            diagnostics.append(f"{text} {row[name]:.2f}")
    if _present(row, "dist_complete"):
        diagnostics.append(f"complete {row['dist_complete']:.0%}")

    grad = (f"grad {row.get('grad_norm', float('nan')):.1f} "
            f"max {row.get('grad_norm_max', float('nan')):.1f} "
            f"clipped {row.get('grad_clipped', 0):.0%}")
    if row.get("batches_skipped"):
        grad += f"  SKIPPED {int(row['batches_skipped'])}"

    header = (f"  [{stamp}] {label}epoch {row['epoch']:3d}/{epochs} {row.get('epoch_seconds', 0):.1f}s")
    loss_line = (f"      loss   train {row['loss']:7.3f}   val {row['val_loss']:7.3f}   best {best_score:7.3f} @{best_epoch}{waiting}")

    return "\n".join((
        header,
        loss_line,
        "      terms  " + "  ".join(terms),
        "      diag   " + "  ".join([*diagnostics, grad]),
    ))

AMP = True


def autocast(enabled=AMP):
    """bfloat16 autocast on hardware that has it, and a no-op everywhere else."""
    usable = (enabled and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=usable)

def _mean_metrics(loader, model, prepare=None):
    """One validation pass, every loss term averaged over rows rather than over batches."""
    totals, rows = {}, 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            count = int(getattr(batch, "num_graphs", 1))
            with autocast():
                measured = model.compute_loss(prepare(batch) if prepare else batch)[1]
            for name, value in measured.items():
                totals[name] = totals.get(name, 0.0) + value * count
            rows += count

    return {name: value / max(rows, 1) for name, value in totals.items()}

def validation_loss(row):
    """The model's own objective on held-out data: the checkpoint criterion."""
    return row["val_loss"]

def selectivity_criterion(row):
    values = [row.get(name) for name in ("val_affinity", "val_selectivity")]
    values = [value for value in values if value is not None and float(value) == float(value)]

    return sum(values) if values else row["val_loss"]

def run_training(model, train_loader, val_loader, optimizer, epochs, grad_clip=10.0,
                    patience=None, scheduler=None, on_epoch_start=None, on_epoch_end=None,
                    report=None, label="", select_on=validation_loss,
                    stop_on=validation_loss, diagnostics=None):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    history = []
    best_score, best_epoch = float("inf"), 0
    waited, best_stop = 0, float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        if on_epoch_start is not None:
            on_epoch_start(epoch)

        started = time.perf_counter()

        model.train()
        totals, batches, norms, skipped = {}, 0, [], 0
        for batch in train_loader:
            with autocast():
                loss, values = model.compute_loss(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()                       # outside autocast: gradients stay fp32

            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))

            if not np.isfinite(norm):
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                continue

            norms.append(norm)
            optimizer.step()

            totals = {k: totals.get(k, 0.0) + v for k, v in values.items()}
            batches += 1

        train_seconds = time.perf_counter() - started
        validation = _mean_metrics(val_loader, model)

        if scheduler is not None:
            scheduler.step()
        if on_epoch_end is not None:
            on_epoch_end(epoch)

        val_loss = validation["loss"]
        norms = norms or [float("nan")]
        history.append({"epoch": epoch, "val_loss": val_loss,
                        **{k: v / max(batches, 1) for k, v in totals.items()},
                        **{f"val_{k}": v for k, v in validation.items()},
                        **(diagnostics(model) if diagnostics is not None else {}),
                        "grad_norm": sum(norms) / len(norms),
                        "grad_norm_max": max(norms),
                        "grad_clipped": sum(n > grad_clip for n in norms) / len(norms),
                        "batches_skipped": skipped,
                        "seconds": train_seconds,
                        "epoch_seconds": time.perf_counter() - started})

        score = select_on(history[-1])
        history[-1]["select_score"] = score
        history[-1]["selectivity_score"] = selectivity_criterion(history[-1])


        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        value = stop_on(history[-1])
        if value < best_stop:
            best_stop, waited = value, 0
        else:
            waited += 1

        if report is None:
            print(format_epoch(history[-1], epochs, best_score, best_epoch, waited, patience, label=label))
        else:
            report(epoch, history[-1], best_score, best_epoch)

        if patience is not None and waited >= patience:
            print(f"  {label}early stop at epoch {epoch}, "
                    f"best {best_score:.4f} @ {best_epoch}")
            break

    if epochs >= 1 and best_epoch == 0:
        raise RuntimeError(f"{label}no epoch produced a finite selection score, so there is no best checkpoint to evaluate; check batches_skipped in the history")

    model.load_state_dict(best_state)

    return history, best_state, best_epoch, best_score

def token_swap(table):
    """A `prepare` hook giving one arm its own tokenisation of the shared batch."""
    cached = {}

    def prepare(batch):
        device = batch.drug_id.device
        if device not in cached:
            cached[device] = table.to(device)

        batch.drug_tokens = cached[device][batch.drug_id.view(-1)]

        return batch

    return prepare

def run_training_concurrently(runs, train_loader, val_loader, epochs, grad_clip=10.0,
                                patience=None, select_on=validation_loss,
                                stop_on=validation_loss, label=""):
    """Every arm steps on the same batches, so the input pipeline is paid for once."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prepared = [run["label"] for run in runs if run.get("prepare") is not None]
    if prepared and len(prepared) != len(runs):
        bare = [run["label"] for run in runs if run.get("prepare") is None]
        raise ValueError(
            f"arms {bare} have no `prepare` while {prepared} do. The arms share one batch "
            f"and the hooks run in order, so these would train on whatever the arms before "
            f"them left behind. Give every arm a prepare, or none.")

    state = {run["label"]: {"run": run, "history": [],
                            "best_score": float("inf"), "best_epoch": 0,
                            "best_stop": float("inf"), "waited": 0,
                            "best_state": {k: v.detach().cpu().clone()
                                            for k, v in run["model"].state_dict().items()}}
                for run in runs}

    active = [run["label"] for run in runs]

    for epoch in range(1, epochs + 1):
        for name in active:
            run = state[name]["run"]
            if run.get("on_epoch_start") is not None:
                run["on_epoch_start"](epoch)
            run["model"].train()

        started = time.perf_counter()
        totals = {name: {} for name in active}
        counts = {name: 0 for name in active}
        norms = {name: [] for name in active}
        skipped = {name: 0 for name in active}

        for batch in train_loader:
            for name in active:
                run = state[name]["run"]
                model, optimizer = run["model"], run["optimizer"]
                prepare = run.get("prepare")

                with autocast():
                    loss, values = model.compute_loss(prepare(batch) if prepare else batch)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))

                if not np.isfinite(norm):
                    optimizer.zero_grad(set_to_none=True)
                    skipped[name] += 1
                    continue

                norms[name].append(norm)
                optimizer.step()

                totals[name] = {k: totals[name].get(k, 0.0) + v for k, v in values.items()}
                counts[name] += 1

        shared = (time.perf_counter() - started) / max(len(active), 1)
        retiring = []

        for name in active:
            arm = state[name]
            run = arm["run"]
            validation = _mean_metrics(val_loader, run["model"], run.get("prepare"))
            arm_norms = norms[name] or [float("nan")]

            row = {"epoch": epoch, "val_loss": validation["loss"],
                    **{k: v / max(counts[name], 1) for k, v in totals[name].items()},
                    **{f"val_{k}": v for k, v in validation.items()},
                    **(run["diagnostics"](run["model"])
                        if run.get("diagnostics") is not None else {}),
                    "grad_norm": sum(arm_norms) / len(arm_norms),
                    "grad_norm_max": max(arm_norms),
                    "grad_clipped": sum(n > grad_clip for n in arm_norms) / len(arm_norms),
                    "batches_skipped": skipped[name],
                    "seconds": shared,
                    "epoch_seconds": time.perf_counter() - started}

            if run.get("on_epoch_end") is not None:
                run["on_epoch_end"](epoch)
            if run.get("scheduler") is not None:
                run["scheduler"].step()

            score = select_on(row)
            row["select_score"] = score
            row["selectivity_score"] = selectivity_criterion(row)
            arm["history"].append(row)

            if score < arm["best_score"]:
                arm["best_score"], arm["best_epoch"] = score, epoch
                arm["best_state"] = {k: v.detach().cpu().clone() for k, v in run["model"].state_dict().items()}

            value = stop_on(row)
            if value < arm["best_stop"]:
                arm["best_stop"], arm["waited"] = value, 0
            else:
                arm["waited"] += 1

            print(format_epoch(row, epochs, arm["best_score"], arm["best_epoch"], arm["waited"], patience, label=f"{label}{name} "))

            if patience is not None and arm["waited"] >= patience:
                print(f"  {label}{name} retired at epoch {epoch}, best {arm['best_score']:.4f} @ {arm['best_epoch']}")
                retiring.append(name)

        for name in retiring:
            active.remove(name)

        if not active:
            break

    results = {}
    for name, arm in state.items():
        if epochs >= 1 and arm["best_epoch"] == 0:
            raise RuntimeError(
                f"{label}{name}: no epoch produced a finite selection score, so there is "
                f"no best checkpoint to evaluate; check batches_skipped in the history")

        arm["run"]["model"].load_state_dict(arm["best_state"])
        results[name] = (arm["history"], arm["best_state"], arm["best_epoch"], arm["best_score"])

    return results

def load_weights(model, checkpoint, device="cpu"):
    """A built model carrying its checkpoint's weights, in eval mode."""
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])

    return model.eval()

def panel_loader(mols, panel_ids, loader, tokenizer, fingerprint_pca=None, batch_size=256, device="cpu"):
    """One batched loader over these molecules alone, built once."""
    return build_generated_loader(
        mols, panel_ids[:1], loader.protein_table, loader.protein_embs, tokenizer,
        fingerprint_pca=fingerprint_pca, batch_size=batch_size, device=device)

def score_with(generated, panel_ids, oracle, loader, device="cpu"):
    """[molecules, targets] of oracle affinity: one pass per target over the same batches."""
    columns = []

    with torch.no_grad():
        for protein_id in panel_ids:
            sequence = loader.protein_table[protein_id].to(device)
            embedding = loader.protein_embs[protein_id].to(device)

            predictions = []
            for batch in generated:
                rows = batch.num_graphs
                batch.prot_seq_cat = sequence.unsqueeze(0).expand(rows, -1)
                batch.prot_emb = embedding.unsqueeze(0).expand(rows, -1)
                predictions.append(oracle(batch).flatten().cpu())

            columns.append(torch.cat(predictions))

    return torch.stack(columns, dim=1)

def atom_counts(loader):
    """How many atoms the training molecules have, as the size distribution to draw from."""
    counts = np.array([int(point.num_nodes) for point in loader.dataset], dtype=np.int64)

    if not len(counts):
        raise ValueError("cannot draw molecule sizes from an empty training loader")

    return counts


def generate_libraries(model, loader, panel_ids, n_generated=256, temperature=1.0, seed=0,
                        device="cpu", size_loader=None):
    """One library per target, generated once so many oracles can score the same sets."""
    if size_loader is None:
        raise ValueError("size_loader is required and must be the training loader")

    model.eval()

    sizes = np.random.default_rng(seed).choice(atom_counts(size_loader), n_generated)

    return [model.generate(
                loader.protein_embs[protein_id].to(device).expand(n_generated, -1),
                loader.protein_table[protein_id].to(device).expand(n_generated, -1),
                seed=seed, temperature=temperature,
                num_atoms=torch.as_tensor(sizes, device=device))
            for protein_id in panel_ids]

def save_libraries(libraries, panel_ids, directory):
    """The generated SMILES themselves, so a new metric never costs another training run."""
    from .chemical_tools import safe_smiles

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for position, (library, protein_id) in enumerate(zip(libraries, panel_ids)):
        for branch, mols in library.items():
            for index, mol in enumerate(mols):
                rows.append({"target": position, "protein_id": int(protein_id),
                                "branch": branch, "index": index,
                                "smiles": safe_smiles(mol, canonical=False) or ""})

    pd.DataFrame(rows).to_csv(directory / "generated.csv", index=False)

def _keep_mask(kept_smiles, known_smiles):
    """Row mask over the affinity matrix: first occurrence, and not a training molecule."""
    seen = set()
    mask = []

    for text in kept_smiles:
        fresh = text not in seen
        seen.add(text)
        mask.append(fresh and (known_smiles is None or text not in known_smiles))

    return mask


def _score_oracle(oracle, generated_loader, panel_ids, loader, device, cache=None):
    """(affinities, per-member matrices, on-target disagreement) for one oracle."""
    cache = {} if cache is None else cache

    def scored(module):
        if module not in cache:
            cache[module] = score_with(generated_loader, panel_ids, module, loader, device)
        return cache[module]

    members = getattr(oracle, "oracles", None)

    if members is None:
        return scored(oracle), {}, None

    names = [member.__class__.__name__ for member in members]
    if len(set(names)) != len(names):
        names = [f"{name}_{index}" for index, name in enumerate(names)]

    parts = {name: scored(member) for name, member in zip(names, members)}
    stacked = torch.stack(list(parts.values()), dim=0)

    return stacked.mean(0), parts, stacked


def score_libraries(libraries, tokenizer, loader, metrics, oracles, panel_ids,
                    reference_mols, fingerprint_pca=None, success_threshold=7.0,
                    reference_activations=None, device="cpu", save_to=None, skip_fcd=False,
                    known_smiles=None, save_scores=True):
    """Every metric per target, averaged over targets, with the libraries written alongside."""
    if save_to is not None:
        save_libraries(libraries, panel_ids, save_to)

    branches = sorted({branch for library in libraries for branch in library})

    # computed once for the whole panel, not rebuilt inside each eval_all call
    panel_gaps = {branch: metrics.tanimoto_gaps([library.get(branch) or [] for library in libraries])
                    for branch in branches}
    reference_mean_atoms = metrics.mean_atoms(reference_mols)
    per_target = []
    matrices = {}
    scored_index = []      # which molecule each affinity-matrix row belongs to

    for position, library in enumerate(libraries):

        for branch in branches:
            generated = library.get(branch)
            if generated is None:
                continue

            row = {"target": position, "protein_id": int(panel_ids[position]), "branch": branch}
            row.update({name: float("nan") for name in LIBRARY_METRICS})
            row.update(metrics.eval_all(
                reference_mols, generated,
                comparison_sets=(),
                known_smiles=known_smiles))
            row["tanimoto_gap"] = panel_gaps[branch][position]
            row["reference_mean_atoms"] = reference_mean_atoms
            row["mean_atoms_ratio"] = (
                row["mean_atoms"] / reference_mean_atoms
                if np.isfinite(row["mean_atoms"])
                and np.isfinite(reference_mean_atoms)
                and reference_mean_atoms > 0
                else float("nan"))

            # built once, scored by every oracle: the build is a loop ove molecules x targets
            generated_loader, kept_smiles = panel_loader(
                generated, panel_ids, loader, tokenizer,
                fingerprint_pca=fingerprint_pca, device=device)

            keep_row = _keep_mask(kept_smiles, known_smiles)
            scored_keep_row = _keep_mask(kept_smiles, None)
            scored_index.extend(
                {"branch": branch, "target": position, "row": index, "smiles": text,
                    "counted": bool(flag)}
                for index, (text, flag) in enumerate(zip(kept_smiles, keep_row)))

            scored_once = {}                  # this library, scored once per oracle
            for oracle_name, oracle in oracles.items():
                block = {name: float("nan") for name in ORACLE_METRICS}
                block.update(success_rate=0.0, effective_delta=0.0, effective_count=0,
                                hit_count=0, scored_count=len(kept_smiles))

                if generated_loader is not None:
                    affinities, parts, stacked = _score_oracle(
                        oracle, generated_loader, panel_ids, loader, device,
                        cache=scored_once)

                    block.update(metrics.eval_affinities(
                        affinities, position, success_threshold, keep=keep_row,
                        total_count=len(generated), scored_keep=scored_keep_row))
                    if stacked is not None:
                        block["oracle_disagreement"] = float(
                            stacked[:, :, position].std(dim=0, unbiased=False).mean())

                    for label, matrix in {**parts,
                                            oracle_name or "MeanOracle": affinities}.items():
                        matrices[f"{branch}|{label}|{position}"] = (
                            matrix.numpy().astype(np.float32))

                prefix = f"{oracle_name}_" if oracle_name else ""
                row.update({prefix + name: value for name, value in block.items()})

            per_target.append(row)

    detail = pd.DataFrame(per_target)
    # save_scores False keeps generated.csv alone: only e6's matrices are ever read back
    if save_to is not None and save_scores and matrices:
        np.savez_compressed(Path(save_to) / "affinities.npz", panel_ids=np.asarray([int(i) for i in panel_ids]), **matrices)

    if save_to is not None and save_scores and scored_index:
        pd.DataFrame(scored_index).to_csv(Path(save_to) / "scored_index.csv", index=False)

    results = {}
    for branch in branches:
        rows = detail[detail["branch"] == branch] if not detail.empty else detail
        if rows.empty:
            continue

        frame = rows.drop(columns=["target", "protein_id", "branch"])
        numeric = frame.mean(numeric_only=True)
        # mean() skips a nan target, so this count is not the count behind every column
        counts = frame[numeric.index].notna().sum()

        results.update({f"{branch}_{name}": float(value) for name, value in numeric.items()})
        # only where a target dropped out, so the CSV does not carry a column of constants

        results.update({f"{branch}_{name}_n": int(counts[name]) for name in numeric.index if counts[name] != len(rows)})
        results[f"{branch}_n_targets"] = len(rows)

        pooled = [mol for library in libraries for mol in (library.get(branch) or [])]
        pooled = metrics.uniqueness(metrics.validity(pooled)[1])[1]
        results[f"{branch}_fcd"] = (
            metrics.frechet_chemnet_distance(pooled, reference_mols, reference=reference_activations)
            if pooled and not skip_fcd else float("nan"))

    return results

def evaluate_generator(model, tokenizer, loader, metrics, oracle, panel_ids, reference_mols,
                        fingerprint_pca=None, n_generated=256, n_eval_targets=None,
                        success_threshold=7.0, temperature=1.0, reference_activations=None,
                        seed=0, device="cpu", save_to=None, known_smiles=None,
                        size_loader=None, skip_fcd=False, save_scores=True):
    """Every metric per decoder branch, averaged over the targets generated for."""
    targets = panel_ids if n_eval_targets is None else panel_ids[:n_eval_targets]
    libraries = generate_libraries(model, loader, targets, n_generated, temperature, seed, device, size_loader=size_loader)

    return score_libraries(libraries, tokenizer, loader, metrics, {"": oracle},
                            panel_ids, reference_mols, fingerprint_pca=fingerprint_pca,
                            success_threshold=success_threshold,
                            reference_activations=reference_activations, device=device,
                            save_to=save_to, known_smiles=known_smiles, skip_fcd=skip_fcd,
                            save_scores=save_scores)
