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
from utils.experiment_tools import (BRANCH_AGGREGATION, DECISION_METRIC,
                                    SUCCESS_THRESHOLD, branch_decision, cold_targets,
                                    collect_runs, complete_blocks, describe_arms,
                                    env_seeds, evaluate_generator, experiment_dir,
                                    finish_run, grid_complete, load_oracle,
                                    matched_init, max_atoms, measured_no_bond_weight,
                                    plot_decision, plot_metric_boxes, record_run,
                                    reference_molecules, run_dir,
                                    run_training_concurrently, save_config,
                                    setup_compute, training_smiles)
from utils.metrics import Metrics
from utils.selectivity import build_objective
from utils.stats_tools import block_design, blocked_tukey_cld

DATASET = "kiba"
USE_SELFIES = True

SEEDS = env_seeds("E2_SEEDS", range(5))
BRANCHES = ["gnn", "seq", "both"]
DECODERS = ["gnn", "seq"]
ARMS = [f"{encoder}->{decoder}" for encoder in BRANCHES for decoder in DECODERS]

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

LOSS_WEIGHTS = dict(w_seq=4.0, w_graph=1.0, w_kl_mean=0.02, w_kl_var=1.0,
                    w_affinity=1.0, w_selectivity=1.0,
                    w_distributional=1.0, affinity_rank_margin=10.0,
                    selectivity_rank_margin=10.0,
                    distribution_mean_margin=10.0, distribution_effect_margin=2.0)

EXPERIMENT = "e2_branch"
CONCURRENT_ARMS = 6


def branch_flags(encoder, decoder):
    return dict(gnn_encoder=encoder in ("gnn", "both"),
                seq_encoder=encoder in ("seq", "both"),
                gnn_decoder=decoder in ("gnn", "both"),
                seq_decoder=decoder in ("seq", "both"))


def main():
    device = setup_compute()
    metrics = Metrics()
    dataset = DATASET
    output = experiment_dir(dataset, EXPERIMENT)
    save_config(output, sys.modules[__name__])
    records, finished = collect_runs(output, ("seed", "arm"), valid={"seed": SEEDS, "arm": ARMS})

    if grid_complete(finished, [{"seed": seed, "arm": arm} for seed in SEEDS for arm in ARMS], "runs"):
        # nothing to train
        frame = pd.DataFrame(records)
        write_analysis(frame, output)

        return frame

    tokenizer = load_tokenizer(dataset, use_selfies=USE_SELFIES)
    shared = shared_drug_ids(dataset, (
        load_tokenizer(dataset, use_selfies=False),
        load_tokenizer(dataset, use_selfies=True)))
    oracle = load_oracle(dataset)
    fingerprint_pca = load_fingerprint_pca(dataset)

    for seed in SEEDS:
        if all(finished(seed=seed, arm=arm) for arm in ARMS):
            continue                              # every arm of this seed already ran

        torch.manual_seed(seed)

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

        pending = [(encoder, decoder) for encoder in BRANCHES for decoder in DECODERS
                    if not finished(seed=seed, arm=f"{encoder}->{decoder}")]

        torch.manual_seed(seed)
        reference_model = build_selvaegen(
            tokenizer, protein_dim, nodes, smiles_len,
            criterion=build_objective(
                train_table, train_loader.protein_embs,
                len(train_loader.drug_token_table), len(train_loader.protein_table),
                pad_id=tokenizer.pad_id, panel_size=PANEL_SIZE, seed=seed, device=device,
                freeze_panel=FREEZE_PANEL, val_table=val_table,
                no_bond_weight=no_bond_weight, **LOSS_WEIGHTS),
            fingerprint_dim=fingerprint_dim, **branch_flags("both", "both"))
        reference_state = {k: v.cpu().clone() for k, v in reference_model.state_dict().items()}
        del reference_model

        # all pending arms train on the same batches
        for group in [pending[i:i + CONCURRENT_ARMS] for i in range(0, len(pending), CONCURRENT_ARMS)]:
            arms = []
            for encoder, decoder in group:
                torch.manual_seed(seed)          # same starting stream for every arm

                objective = build_objective(
                    train_table, train_loader.protein_embs,
                    len(train_loader.drug_token_table), len(train_loader.protein_table),
                    pad_id=tokenizer.pad_id, panel_size=PANEL_SIZE, seed=seed,
                    device=device, freeze_panel=FREEZE_PANEL,
                    val_table=val_table, no_bond_weight=no_bond_weight, **LOSS_WEIGHTS)

                model = build_selvaegen(
                    tokenizer, protein_dim, nodes, smiles_len, criterion=objective,
                    fingerprint_dim=fingerprint_dim,
                    **branch_flags(encoder, decoder)).to(device)

                def start(epoch, model=model, objective=objective):
                    model.set_protein_prior_trainable(epoch > FREEZE_PRIOR_EPOCHS)
                    objective.set_epoch(epoch)

                print(f"\n{dataset} | seed {seed} | {encoder}->{decoder} | {sum(p.numel() for p in model.parameters()):,} parameters")

                arms.append({"label": f"{encoder}->{decoder}", "model": model,
                                "optimizer": torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
                                "on_epoch_start": start,
                                "diagnostics": objective.geometry,
                                "objective": objective,
                                "encoder": encoder, "decoder": decoder})

            matched_init([arm["model"] for arm in arms], reference=reference_state)

            reset_loader_epoch(train_loader)
            started = time.perf_counter()
            trained = run_training_concurrently(
                arms, train_loader, val_loader, EPOCHS, grad_clip=GRAD_CLIP,
                patience=PATIENCE, label=f"seed{seed} ")
            elapsed = time.perf_counter() - started

            for arm in arms:
                encoder, decoder = arm["encoder"], arm["decoder"]
                model = arm["model"]
                history, best_state, best_epoch, best_loss = trained[arm["label"]]

                stem = run_dir(dataset, EXPERIMENT, f"seed{seed}_{encoder}-{decoder}")
                summary = finish_run(history, model, elapsed / len(arms), stem, grad_clip=GRAD_CLIP, state=best_state)

                scores = evaluate_generator(
                    model, tokenizer, val_loader, metrics, oracle, panel_ids, reference,
                    fingerprint_pca=fingerprint_pca, n_generated=N_GENERATED,
                    n_eval_targets=N_EVAL_TARGETS,
                    success_threshold=SUCCESS_THRESHOLD[dataset],
                    reference_activations=activations, seed=seed, device=device,
                    known_smiles=known, size_loader=train_loader)

                records.append({"dataset": dataset, "seed": seed,
                                "split_seed": seed, "evaluation_split": "validation",
                                "tokens": "selfies" if USE_SELFIES else "smiles",
                                "encoder": encoder,
                                "decoder": decoder, "arm": f"{encoder}->{decoder}",
                                "best_epoch": best_epoch, "best_val_loss": best_loss,
                                **summary, **history[best_epoch - 1], **scores})
                record_run(stem, records[-1])
                records, finished = collect_runs(output, ("seed", "arm"), valid={"seed": SEEDS, "arm": ARMS})

            for arm in arms:
                del arm["model"], arm["optimizer"], arm["objective"]
            arms.clear()
            torch.cuda.empty_cache()

        write_analysis(pd.DataFrame(records), output)

    return pd.DataFrame(records)

def compare(frame):
    """One confirmatory number per arm: effective_delta over the branches it has."""
    rows = []

    for dataset, block in frame.groupby("dataset"):
        block = block.copy()
        block[DECISION_METRIC] = branch_decision(block)

        values, names, _ = block_design(block, "arm", DECISION_METRIC)
        if values.shape[1] < 2 or values.shape[0] < 2:
            continue

        for name, result in blocked_tukey_cld(values, names).items():
            rows.append({"dataset": dataset, "metric": DECISION_METRIC,
                            "aggregation": BRANCH_AGGREGATION, "family": "primary",
                            "arm": name, **result})

    return pd.DataFrame(rows)


def write_analysis(frame, output):
    """Tukey table, summary and figures for whatever seeds are complete."""
    if frame.empty:
        return pd.DataFrame(), []

    frame = frame[frame["arm"].isin(ARMS)]
    # summary, figures and tests all describe the same seeds
    frame = complete_blocks(frame, "arm")
    # the one confirmatory test
    comparison = compare(frame)
    comparison.to_csv(output / "tukey.csv", index=False)

    # everything else
    describe_arms(frame, "arm", output / "summary.csv")

    frame = frame.copy()
    frame[DECISION_METRIC] = branch_decision(frame)
    plot_decision(frame, "arm", output / "figures" / "decision.pdf")
    figures = plot_metric_boxes(frame, "arm", output / "figures", order=ARMS,
                                merge_branches=True)

    return comparison, figures


if __name__ == "__main__":
    frame = main()
    output = experiment_dir(DATASET, EXPERIMENT)
    comparison, figures = write_analysis(frame, output)

    print(f"\n=== branch configurations, Tukey on {DECISION_METRIC} ({BRANCH_AGGREGATION} over branches) ===")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n{len(figures) + 1} figures and summary.csv written to {output}")
