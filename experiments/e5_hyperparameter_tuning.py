import json
import math
import os
import sys
from pathlib import Path

import optuna
import torch
from rdkit import Chem

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models.sel_vae_gen import build_selvaegen
from utils.data_tools import (create_dataloaders, load_fingerprint_pca, load_tokenizer,
                            reset_loader_epoch, shared_drug_ids)
from utils.experiment_tools import (SUCCESS_THRESHOLD, cold_targets, evaluate_generator,
                                    experiment_dir, load_oracle, max_atoms,
                                    measured_no_bond_weight, reference_molecules,
                                    run_training, save_config, setup_compute,
                                    training_smiles)
from utils.metrics import Metrics
from utils.selectivity import build_objective

DATASET = "kiba"
USE_SELFIES = True

SEED = 0
TRIALS = 100
EPOCHS = 50
PATIENCE = 15

# dual gpu
DEVICES = ["cuda:0", "cuda:1"] if torch.cuda.device_count() > 1 else [None]

BATCH_SIZE = 256
GROUPS_PER_BATCH = 16
GRAD_CLIP = 50.0
PANEL_SIZE = 3
FREEZE_PRIOR_EPOCHS = 10
FREEZE_PANEL = False

N_GENERATED = 256
N_REFERENCE = 2000
N_PANEL_TARGETS = None
N_EVAL_TARGETS = None          # every cold target, not the best-measured ones

BRANCH_FLAGS = dict(gnn_encoder=True, seq_encoder=True,
                    gnn_decoder=False, seq_decoder=True)
LOSS_WEIGHTS = dict(w_affinity=1.0, w_selectivity=1.0, w_distributional=1.0)
ACTIVE_BRANCHES = tuple(branch for branch, enabled in
                        (("seq", BRANCH_FLAGS["seq_decoder"]),
                            ("graph", BRANCH_FLAGS["gnn_decoder"])) if enabled)

MIN_VALIDITY = 0.10
MIN_MEAN_ATOMS_RATIO = 0.50
INVALID_SCORE = -1.0e9
REPORTED_METRICS = ("validity", "mean_atoms", "reference_mean_atoms", "mean_atoms_ratio",
                    "delta_score", "selectivity_effect", "directional_consistency",
                    "success_rate", "tanimoto_gap", "hit_count", "scored_count",
                    "hit_delta_score", "effective_delta")

EXPERIMENT = "e5_tuning"


def branch_value(scores, metric, branch):
    try:
        return float(scores.get(f"{branch}_{metric}", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def branch_score(scores, branch):
    """(cleared the gates, objective) for one branch."""
    validity = branch_value(scores, "validity", branch)
    if not math.isfinite(validity):
        return False, INVALID_SCORE
    if validity < MIN_VALIDITY:
        return False, INVALID_SCORE + max(validity, 0.0)

    size_ratio = branch_value(scores, "mean_atoms_ratio", branch)
    if not math.isfinite(size_ratio):
        return False, INVALID_SCORE
    if size_ratio < MIN_MEAN_ATOMS_RATIO:
        return False, INVALID_SCORE + max(size_ratio, 0.0) / MIN_MEAN_ATOMS_RATIO

    score = branch_value(scores, "effective_delta", branch)
    if not math.isfinite(score):
        return False, INVALID_SCORE

    return True, score


def objective_score(scores):
    """(every active branch cleared its gates, mean objective over them)."""
    results = [branch_score(scores, branch) for branch in ACTIVE_BRANCHES]

    if not results:
        return False, INVALID_SCORE

    return (all(cleared for cleared, _ in results), sum(value for _, value in results) / len(results))


def best_cleared_trial(study):
    """The best trial that cleared every gate, or None if no trial did."""
    cleared = [trial for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None
                and trial.user_attrs.get("passed_gates", False)]

    return max(cleared, key=lambda trial: trial.value) if cleared else None


def search_space(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        
        fusion_dim=trial.suggest_categorical("fusion_dim", [128, 256, 512]),
        
        w_seq=trial.suggest_float("w_seq", 0.01, 5.0, log=True),
        w_kl_mean=trial.suggest_float("w_kl_mean", 0.001, 0.5, log=True),
        w_kl_var=trial.suggest_float("w_kl_var", 0.01, 5.0, log=True),
        
        direction_weight=trial.suggest_float("direction_weight", 0.0, 1.0),
        affinity_rank_margin=trial.suggest_float("affinity_rank_margin", 1.0, 50.0, log=True),
        selectivity_rank_margin=trial.suggest_float("selectivity_rank_margin", 1.0, 50.0, log=True),
        distribution_mean_margin=trial.suggest_float("distribution_mean_margin", 0.5, 20.0, log=True),
        distribution_effect_margin=trial.suggest_float("distribution_effect_margin", 0.2, 3.0, log=True),
    )


def kept_trials(study):
    """Everything the budget counts: a failed trial is retried, not spent."""
    return [trial for trial in study.trials if trial.state != optuna.trial.TrialState.FAIL]

def open_study(output):
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{(output / 'study.db').as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 120}})

    sampler = optuna.samplers.TPESampler(seed=42)

    return optuna.create_study(
        study_name="e5",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner())

def main(device=None):
    default = setup_compute()
    device = default if device is None else device
    if ":" in str(device):
        torch.cuda.set_device(device)

    output = experiment_dir(DATASET, EXPERIMENT)
    print(f"  explicit config | tokens {'selfies' if USE_SELFIES else 'smiles'} | branches {ACTIVE_BRANCHES} | loss weights {LOSS_WEIGHTS}")

    metrics = Metrics()
    tokenizers = {"smiles": load_tokenizer(DATASET, use_selfies=False),
                    "selfies": load_tokenizer(DATASET, use_selfies=True)}
    shared = shared_drug_ids(DATASET, tokenizers.values())
    tokenizer = tokenizers["selfies" if USE_SELFIES else "smiles"]
    oracle = load_oracle(DATASET)
    fingerprint_pca = load_fingerprint_pca(DATASET)

    (train_loader, val_loader, _), (train_table, val_table, _) = create_dataloaders(
        DATASET, tokenizer=tokenizer, batch_size=BATCH_SIZE, train_frac=0.8, val_frac=0.1,
        test_frac=0.1, random_state=SEED, device=device,
        groups_per_batch=GROUPS_PER_BATCH, restrict_drug_ids=shared,
        with_protein_sequence=False)

    panel_ids = cold_targets(val_loader)[:N_PANEL_TARGETS]
    reference = reference_molecules(train_loader, N_REFERENCE)
    known = training_smiles(train_loader)
    activations = metrics.chemnet_activations([Chem.MolToSmiles(mol) for mol in reference if mol is not None])

    protein_dim = train_loader.protein_embs.shape[1]
    fingerprint_dim = train_loader.fingerprint_table.shape[1]
    nodes = max_atoms(train_loader, val_loader)
    smiles_len = train_loader.drug_token_table.shape[1]
    no_bond_weight = measured_no_bond_weight(train_loader)
    print(f"  no_bond_weight {no_bond_weight:.4f} (measured on this split)")

    def run(trial):
        torch.manual_seed(SEED)
        reset_loader_epoch(train_loader)
        space = search_space(trial)
        learning_rate = space.pop("learning_rate")
        weight_decay = space.pop("weight_decay")
        fusion_dim = space.pop("fusion_dim")

        criterion = build_objective(
            train_table, train_loader.protein_embs, len(train_loader.drug_token_table),
            len(train_loader.protein_table), pad_id=tokenizer.pad_id, panel_size=PANEL_SIZE,
            seed=SEED, device=device, freeze_panel=FREEZE_PANEL,
            val_table=val_table, no_bond_weight=no_bond_weight, **LOSS_WEIGHTS, **space)

        model = build_selvaegen(
            tokenizer, protein_dim, nodes, smiles_len, criterion=criterion,
            fusion_dim=fusion_dim, fingerprint_dim=fingerprint_dim, **BRANCH_FLAGS).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        def start(epoch):
            model.set_protein_prior_trainable(epoch > FREEZE_PRIOR_EPOCHS)
            criterion.set_epoch(epoch)

        def report(epoch, row, best_loss, best_epoch):
            print(f"  trial {trial.number:3d} epoch {epoch:3d} | val_loss {row['val_loss']:.3f} | best {best_loss:.3f} @ {best_epoch}")
            loss = float(row["val_loss"])
            trial.set_user_attr(f"val_loss_epoch_{epoch}", loss)
            trial.report(loss, epoch)

            if not math.isfinite(loss):
                raise optuna.TrialPruned()

        try:
            _, _, best_epoch, _ = run_training(
                model, train_loader, val_loader, optimizer, EPOCHS, grad_clip=GRAD_CLIP,
                patience=PATIENCE, on_epoch_start=start, report=report,
                diagnostics=criterion.geometry)
            trial.set_user_attr("best_epoch", best_epoch)

            scores = evaluate_generator(
                model, tokenizer, val_loader, metrics, oracle, panel_ids, reference,
                fingerprint_pca=fingerprint_pca, n_generated=N_GENERATED,
                n_eval_targets=N_EVAL_TARGETS,
                success_threshold=SUCCESS_THRESHOLD[DATASET],
                reference_activations=activations, seed=SEED, device=device,
                skip_fcd=True,
                known_smiles=known, size_loader=train_loader)
        finally:
            model = criterion = optimizer = None
            torch.cuda.empty_cache()

        for branch in ACTIVE_BRANCHES:
            cleared, value = branch_score(scores, branch)
            trial.set_user_attr(f"{branch}_score", value)
            trial.set_user_attr(f"{branch}_passed_gates", cleared)
            for metric in REPORTED_METRICS:
                trial.set_user_attr(f"{branch}_{metric}", branch_value(scores, metric, branch))

        cleared, value = objective_score(scores)
        trial.set_user_attr("passed_gates", cleared)

        return value

    study = open_study(output)

    def checkpoint(study, _trial=None):
        # every worker writes this
        staging = output / f"trials.csv.{os.getpid()}.tmp"
        study.trials_dataframe().to_csv(staging, index=False)
        os.replace(staging, output / "trials.csv")

        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        
        diverged = sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials)
        if not completed:
            return

        best = best_cleared_trial(study)

        if best is None:
            print(f"  {len(completed)} trials complete, {diverged} diverged, none passed the gates. best_params.json not written")
            return

        (output / "best_params.json").write_text(json.dumps(best.params, indent=2), encoding="utf-8")
        print(f"  best so far: trial {best.number} = {best.value:.4f} "
                f"({len(completed)} complete and {diverged} diverged of {TRIALS}, "
                f"{sum(t.user_attrs.get('passed_gates', False) for t in completed)} past the gates)")

    def stop_when_full(study, _trial=None):
        if len(kept_trials(study)) >= TRIALS:
            study.stop()

    done = len(kept_trials(study))
    print(f"  {device}: {done} trials in the study, target {TRIALS}")

    try:
        if done < TRIALS:
            study.optimize(run, n_trials=None, callbacks=[checkpoint, stop_when_full])
    except KeyboardInterrupt:
        print("\ninterrupted -- rerunning resumes from the same study database")

    checkpoint(study)
    return study


def search():
    """One worker per device, all sharing the study, or a single in-process run."""
    # both before any worker exists: CREATE TABLE races, and config.json truncates
    output = experiment_dir(DATASET, EXPERIMENT)
    open_study(output)
    save_config(output, sys.modules[__name__])

    if len(DEVICES) == 1:
        return main(DEVICES[0])

    context = torch.multiprocessing.get_context("spawn")
    workers = [context.Process(target=main, args=(device,)) for device in DEVICES]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    failed = [device for device, worker in zip(DEVICES, workers) if worker.exitcode]
    if failed:
        raise SystemExit(f"the worker(s) on {failed} exited non-zero; the study database keeps every finished trial, so rerunning resumes")

    return open_study(experiment_dir(DATASET, EXPERIMENT))


if __name__ == "__main__":
    study = search()
    best = best_cleared_trial(study)

    if best is None:
        raise SystemExit(
            "no trial cleared MIN_VALIDITY and MIN_MEAN_ATOMS_RATIO on every active "
            "branch, so there is no configuration to hand to e6. Read trials.csv: the "
            "*_validity and *_mean_atoms_ratio columns say which gate is binding.")

    print(f"\n=== best trial that cleared the gates: {best.number} = {best.value:.4f} ===")
    for name, value in best.params.items():
        print(f"  {name:26s} {value}")
