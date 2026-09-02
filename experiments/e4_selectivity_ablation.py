import sys
import time
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models.sel_vae_gen import build_selvaegen
from utils.data_tools import (create_dataloaders, load_fingerprint_pca, load_tokenizer,
                            reset_loader_epoch, shared_drug_ids)
from utils.experiment_tools import (DECISION_METRIC,
                                    SUCCESS_THRESHOLD, branch_decision, cold_targets,
                                    collect_runs, complete_blocks, describe_arms,
                                    env_seeds, evaluate_generator, experiment_dir,
                                    finish_run, grid_complete, load_oracle, max_atoms,
                                    measured_no_bond_weight, plot_decision,
                                    plot_metric_boxes, record_run, reference_molecules,
                                    run_dir, run_training_concurrently, save_config,
                                    setup_compute, training_smiles)
from utils.metrics import Metrics
from utils.selectivity import build_objective
from utils.stats_tools import adjust_family, block_design, blocked_dunnett

DATASET = "kiba"

SEEDS = env_seeds("E4_SEEDS", range(5))
TERMS = ("affinity", "selectivity", "distributional")
ARMS = ((False, False, False),      # control
        (False, False, True),       # the distributional term alone
        (True, True, False),        # the pairwise ranking terms alone
        (True, True, True))         # everything

EPOCHS = 150
PATIENCE = 30
BATCH_SIZE = 256
GROUPS_PER_BATCH = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 50.0
FREEZE_PRIOR_EPOCHS = 10
PANEL_SIZE = 3
FREEZE_PANEL = False
N_GENERATED = 256
N_REFERENCE = 2000
N_PANEL_TARGETS = None
N_EVAL_TARGETS = None          # every cold target, not the best-measured ones
BASE_WEIGHTS = dict(w_seq=4.0, w_graph=1.0, w_kl_mean=0.02, w_kl_var=1.0,
                    affinity_rank_margin=10.0,
                    selectivity_rank_margin=10.0,
                    distribution_mean_margin=10.0, distribution_effect_margin=2.0)
CONTROL_ARM = "none"

SELECTIVITY_METRICS = ("delta_score", "selectivity_effect", "directional_consistency",
                        "tanimoto_gap", "success_rate", "effective_delta")

EXPERIMENT = "e4_selectivity"
CONCURRENT_ARMS = 4



def arms():
    for enabled in ARMS:
        weights = {f"w_{term}": float(on) for term, on in zip(TERMS, enabled)}
        label = "+".join(term for term, on in zip(TERMS, enabled) if on) or "none"

        yield label, weights

def main():
    device = setup_compute()
    metrics = Metrics()
    dataset = DATASET
    output = experiment_dir(dataset, EXPERIMENT)
    save_config(output, sys.modules[__name__])
    records, finished = collect_runs(output, ("seed", "arm"), valid={"seed": SEEDS})
    arm_labels = [label for label, _ in arms()]

    # checked before the loaders
    if grid_complete(finished, [{"seed": seed, "arm": label} for seed in SEEDS for label in arm_labels], "runs"):
        # nothing to train
        frame = pd.DataFrame(records)
        write_analysis(frame, output)

        return frame

    tokenizer = load_tokenizer(dataset, use_selfies=True)
    shared = shared_drug_ids(dataset, (load_tokenizer(dataset, use_selfies=False), tokenizer))
    oracle = load_oracle(dataset)
    fingerprint_pca = load_fingerprint_pca(dataset)
    arm_grid = list(arms())

    for seed in SEEDS:
        if all(finished(seed=seed, arm=label) for label, _ in arm_grid):
            continue

        (train_loader, val_loader, _), (train_table, val_table, _) = create_dataloaders(
            dataset, tokenizer=tokenizer, batch_size=BATCH_SIZE, train_frac=0.8,
            val_frac=0.1, test_frac=0.1, random_state=seed, device=device,
            groups_per_batch=GROUPS_PER_BATCH, restrict_drug_ids=shared,
            with_protein_sequence=False)

        panel_ids = cold_targets(val_loader)[:N_PANEL_TARGETS]
        nodes = max_atoms(train_loader, val_loader)
        no_bond_weight = measured_no_bond_weight(train_loader)
        print(f"  no_bond_weight {no_bond_weight:.4f} (measured on this split)")
        smiles_len = train_loader.drug_token_table.shape[1]
        protein_dim = train_loader.protein_embs.shape[1]
        fingerprint_dim = train_loader.fingerprint_table.shape[1]
        reference = reference_molecules(train_loader, N_REFERENCE)
        known = training_smiles(train_loader)
        activations = metrics.chemnet_activations([Chem.MolToSmiles(mol) for mol in reference if mol is not None])

        pending = [(label, weights) for label, weights in arm_grid if not finished(seed=seed, arm=label)]

        for group in [pending[i:i + CONCURRENT_ARMS] for i in range(0, len(pending), CONCURRENT_ARMS)]:
            trained_arms = []
            for label, weights in group:
                torch.manual_seed(seed)          # same starting stream for every arm

                objective = build_objective(
                    train_table, train_loader.protein_embs,
                    len(train_loader.drug_token_table), len(train_loader.protein_table),
                    pad_id=tokenizer.pad_id, panel_size=PANEL_SIZE, seed=seed,
                    device=device, freeze_panel=FREEZE_PANEL,
                    val_table=val_table, no_bond_weight=no_bond_weight,
                    **BASE_WEIGHTS, **weights)

                model = build_selvaegen(tokenizer, protein_dim, nodes, smiles_len,
                                        criterion=objective,
                                        fingerprint_dim=fingerprint_dim).to(device)

                def start(epoch, model=model, objective=objective):
                    model.set_protein_prior_trainable(epoch > FREEZE_PRIOR_EPOCHS)
                    objective.set_epoch(epoch)

                print(f"\n{dataset} | seed {seed} | terms: {label}")

                trained_arms.append({"label": label, "model": model,
                                "optimizer": torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
                                "on_epoch_start": start,
                                "diagnostics": objective.geometry,
                                "objective": objective})

            reset_loader_epoch(train_loader)
            started = time.perf_counter()
            trained = run_training_concurrently(
                trained_arms, train_loader, val_loader, EPOCHS, grad_clip=GRAD_CLIP,
                patience=PATIENCE, label=f"seed{seed} ")
            elapsed = time.perf_counter() - started

            for arm in trained_arms:
                arm["stem"] = run_dir(dataset, EXPERIMENT, f"seed{seed}_{arm['label']}")
                arm["trained"] = trained[arm["label"]]
                arm["summary"] = finish_run(
                    arm["trained"][0], arm["model"], elapsed / len(trained_arms),
                    arm["stem"], grad_clip=GRAD_CLIP, state=arm["trained"][1])

            failed = []
            for arm in trained_arms:
                label, stem = arm["label"], arm["stem"]
                history, _, best_epoch, best_loss = arm["trained"]

                try:
                    scores = evaluate_generator(
                        arm["model"], tokenizer, val_loader, metrics, oracle, panel_ids,
                        reference, fingerprint_pca=fingerprint_pca,
                        n_generated=N_GENERATED, n_eval_targets=N_EVAL_TARGETS,
                        success_threshold=SUCCESS_THRESHOLD[dataset],
                        reference_activations=activations, seed=seed, device=device,
                        known_smiles=known, size_loader=train_loader)
                except Exception as error:
                    failed.append(label)
                    print(f"  seed{seed} {label}: scoring failed, weights kept at "
                            f"{stem / 'model.pt'} -- {type(error).__name__}: {error}",
                            flush=True)
                    continue

                records.append({"dataset": dataset, "seed": seed,
                                "split_seed": seed, "evaluation_split": "validation",
                                "tokens": "selfies", "arm": label,
                                "best_epoch": best_epoch, "best_val_loss": best_loss,
                                **arm["summary"], **history[best_epoch - 1], **scores})
                record_run(stem, records[-1])
                records, finished = collect_runs(output, ("seed", "arm"), valid={"seed": SEEDS})

            if failed:
                print(f"  seed{seed}: {len(failed)} of {len(trained_arms)} arms unscored "
                        f"({', '.join(failed)}); the rest were recorded and only these "
                        f"will be retrained", flush=True)

            for arm in trained_arms:
                del arm["model"], arm["optimizer"], arm["objective"]
            trained_arms.clear()
            torch.cuda.empty_cache()

        write_analysis(pd.DataFrame(records), output)

    return pd.DataFrame(records)

DECISION_BRANCHES = ("seq",)
DECISION_AGGREGATION = "sequence"

def compare(frame):
    """One number per arm, Dunnett against the control."""
    contrasts = []

    for dataset, block in frame.groupby("dataset"):
        block = block.copy()
        block[DECISION_METRIC] = branch_decision(
            block, branches=DECISION_BRANCHES)

        values, names, _ = block_design(block, "arm", DECISION_METRIC)
        if values.shape[1] < 2 or values.shape[0] < 2 or CONTROL_ARM not in names:
            continue

        for row in blocked_dunnett(values, names, CONTROL_ARM).to_dict("records"):
            contrasts.append({"dataset": dataset, "metric": DECISION_METRIC,
                                "aggregation": DECISION_AGGREGATION,
                                "family": "primary", **row})

    return adjust_family(pd.DataFrame(contrasts))


def write_analysis(frame, output):
    """Dunnett contrasts, summary and figures for whatever seeds are complete."""
    if frame.empty:
        return pd.DataFrame(), []

    frame = complete_blocks(frame, "arm")
    order = [label for label, _ in arms()]

    contrasts = compare(frame)
    contrasts.to_csv(output / "dunnett.csv", index=False)

    describe_arms(frame, "arm", output / "summary.csv")

    plot_decision(frame, "arm", output / "figures" / "decision.pdf", order=order,
                    metric=DECISION_METRIC, branches=DECISION_BRANCHES,
                    aggregation=DECISION_AGGREGATION)
    figures = plot_metric_boxes(frame, "arm", output / "figures", order=order)

    return contrasts, figures


if __name__ == "__main__":
    frame = main()
    output = experiment_dir(DATASET, EXPERIMENT)
    contrasts, figures = write_analysis(frame, output)

    print(f"\n=== each arm against '{CONTROL_ARM}', blocked Dunnett on {DECISION_METRIC} ({DECISION_AGGREGATION} branch) ===")
    if contrasts.empty:
        print("  no usable contrast yet")
    else:
        print(contrasts.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n{len(figures) + 1} figures and summary.csv written to {output}")
