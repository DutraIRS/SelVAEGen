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
from utils.experiment_tools import (SUCCESS_THRESHOLD, cold_targets, collect_runs,
                                    complete_blocks, describe_arms, env_seeds,
                                    evaluate_generator, experiment_dir, finish_run,
                                    grid_complete, load_oracle, matched_init, max_atoms,
                                    measured_no_bond_weight, plot_decision,
                                    plot_metric_boxes, record_run, reference_molecules,
                                    run_dir, run_training_concurrently, save_config,
                                    setup_compute, token_swap, training_smiles)
from utils.metrics import Metrics
from utils.selectivity import build_objective
from utils.stats_tools import paired_comparison

DATASET = "kiba"

SEEDS = env_seeds("E3_SEEDS", range(5))

TOKEN_TYPES = ["smiles", "selfies"]

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
N_EVAL_TARGETS = None

LOSS_WEIGHTS = dict(w_seq=4.0, w_graph=1.0, w_kl_mean=0.02, w_kl_var=1.0,
                    w_affinity=1.0, w_selectivity=1.0,
                    w_distributional=1.0, affinity_rank_margin=10.0,
                    selectivity_rank_margin=10.0,
                    distribution_mean_margin=10.0, distribution_effect_margin=2.0)

EXPERIMENT = "e3_selfies"
DECISION_METRIC = "seq_validity"
TEST_METRICS = [DECISION_METRIC, "seq_effective_delta"]



def main():
    device = setup_compute()
    metrics = Metrics()
    dataset = DATASET
    output = experiment_dir(dataset, EXPERIMENT)
    save_config(output, sys.modules[__name__])
    records, finished = collect_runs(output, ("seed", "tokens"), valid={"seed": SEEDS})

    # checked before the loaders
    if grid_complete(finished, [{"seed": seed, "tokens": t} for seed in SEEDS for t in TOKEN_TYPES], "runs"):
        # nothing to train
        frame = pd.DataFrame(records)
        write_analysis(frame, output)

        return frame

    oracle = load_oracle(dataset)
    fingerprint_pca = load_fingerprint_pca(dataset)
    tokenizers = {"smiles": load_tokenizer(dataset, use_selfies=False),
                    "selfies": load_tokenizer(dataset, use_selfies=True)}

    shared = shared_drug_ids(dataset, tokenizers.values())
    print(f"{dataset}: {len(shared):,} drugs encodable under both vocabularies")

    for seed in SEEDS:
        if all(finished(seed=seed, tokens=token) for token in TOKEN_TYPES):
            continue                              # both tokenisers of this seed already ran

        reference, activations, known = None, None, None
        reference_state = None       # shared initialisation for both tokenisers

        due = [t for t in TOKEN_TYPES if not finished(seed=seed, tokens=t)]
        arms, built = [], {}

        for token_type in due:
            torch.manual_seed(seed)
            tokenizer = tokenizers[token_type]

            (train_loader, val_loader, _), (train_table, val_table, _) = create_dataloaders(
                dataset, tokenizer=tokenizer, batch_size=BATCH_SIZE, train_frac=0.8,
                val_frac=0.1, test_frac=0.1, random_state=seed, device=device,
                groups_per_batch=GROUPS_PER_BATCH, restrict_drug_ids=shared,
                with_protein_sequence=False)
            reset_loader_epoch(train_loader)

            no_bond_weight = measured_no_bond_weight(train_loader)

            if reference is None:
                reference = reference_molecules(train_loader, N_REFERENCE)
                known = training_smiles(train_loader)
                activations = metrics.chemnet_activations([Chem.MolToSmiles(mol) for mol in reference if mol is not None])

            objective = build_objective(
                train_table, train_loader.protein_embs,
                len(train_loader.drug_token_table), len(train_loader.protein_table),
                pad_id=tokenizer.pad_id, panel_size=PANEL_SIZE, seed=seed,
                device=device, freeze_panel=FREEZE_PANEL,
                val_table=val_table, no_bond_weight=no_bond_weight, **LOSS_WEIGHTS)

            model = build_selvaegen(
                tokenizer, train_loader.protein_embs.shape[1],
                max_atoms(train_loader, val_loader),
                train_loader.drug_token_table.shape[1], criterion=objective,
                fingerprint_dim=train_loader.fingerprint_table.shape[1]).to(device)

            if reference_state is None:
                reference_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                matched_init([model], reference=reference_state)

            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            print(f"\n{dataset} | seed {seed} | {token_type} | {sum(p.numel() for p in model.parameters()):,} parameters")

            def start(epoch, model=model, objective=objective):
                model.set_protein_prior_trainable(epoch > FREEZE_PRIOR_EPOCHS)
                objective.set_epoch(epoch)

            arms.append({"label": token_type, "model": model, "optimizer": optimizer,
                            "on_epoch_start": start,
                            "prepare": token_swap(train_loader.drug_token_table),
                            "diagnostics": objective.geometry})
            built[token_type] = dict(tokenizer=tokenizer, model=model,
                                        objective=objective, optimizer=optimizer,
                                        train_loader=train_loader, val_loader=val_loader)

        if not arms:
            continue

        lead = built[due[0]]
        reset_loader_epoch(lead["train_loader"])
        started = time.perf_counter()
        trained = run_training_concurrently(
            arms, lead["train_loader"], lead["val_loader"], EPOCHS,
            grad_clip=GRAD_CLIP, patience=PATIENCE, label=f"{dataset} seed{seed} ")

        summaries = {}
        for token_type in due:
            made = built[token_type]
            history, best_state = trained[token_type][0], trained[token_type][1]
            stem = run_dir(dataset, EXPERIMENT, f"seed{seed}_{token_type}")
            summaries[token_type] = (stem, finish_run(
                history, made["model"], time.perf_counter() - started, stem,
                grad_clip=GRAD_CLIP, state=best_state))

        failed = []
        for token_type in due:
            made = built[token_type]
            tokenizer, model = made["tokenizer"], made["model"]
            train_loader, val_loader = made["train_loader"], made["val_loader"]
            history, _, best_epoch, best_loss = trained[token_type]
            stem, summary = summaries[token_type]

            try:
                scores = evaluate_generator(
                    model, tokenizer, val_loader, metrics, oracle,
                    cold_targets(val_loader)[:N_PANEL_TARGETS], reference, fingerprint_pca,
                    n_generated=N_GENERATED, n_eval_targets=N_EVAL_TARGETS,
                    success_threshold=SUCCESS_THRESHOLD[dataset],
                    reference_activations=activations, seed=seed, device=device,
                    known_smiles=known, size_loader=train_loader)
            except Exception as error:
                failed.append(token_type)
                print(f"  seed{seed} {token_type}: scoring failed, weights kept at {stem / 'model.pt'} -- {type(error).__name__}: {error}", flush=True)
                continue

            records.append({"dataset": dataset, "seed": seed,
                            "split_seed": seed, "evaluation_split": "validation",
                            "tokens": token_type,
                            "best_epoch": best_epoch, "best_val_loss": best_loss,
                            **summary, **history[best_epoch - 1], **scores})
            record_run(stem, records[-1])
            records, finished = collect_runs(output, ("seed", "tokens"), valid={"seed": SEEDS})

        if failed:
            print(f"  seed{seed}: {len(failed)} of {len(due)} arms unscored "
                    f"({', '.join(failed)}); the rest were recorded and only these will "
                    f"be retrained", flush=True)

        for made in built.values():
            made["model"] = made["objective"] = made["optimizer"] = None
        arms.clear()
        torch.cuda.empty_cache()

        write_analysis(pd.DataFrame(records), output)

    return pd.DataFrame(records)


def write_analysis(frame, output):
    """Paired test, summary and figures for whatever seeds are complete."""
    if frame.empty:
        return pd.DataFrame(), []

    frame = complete_blocks(frame, "tokens")
    tests = paired_comparison(frame, group_column="tokens", baseline="smiles",
                                treatment="selfies", metrics=TEST_METRICS,
                                pair_on="seed", by="dataset")
    tests.to_csv(output / "ttest.csv", index=False)

    describe_arms(frame, "tokens", output / "summary.csv")
    plot_decision(frame, "tokens", output / "figures" / "decision.pdf",
                    order=TOKEN_TYPES, metric=("validity", "effective_delta"),
                    branches=("seq",), aggregation="sequence")
    figures = plot_metric_boxes(frame, "tokens", output / "figures", order=TOKEN_TYPES)

    return tests, figures


if __name__ == "__main__":
    frame = main()
    output = experiment_dir(DATASET, EXPERIMENT)
    tests, figures = write_analysis(frame, output)

    print(f"\n=== SELFIES vs SMILES, paired over seeds on {DECISION_METRIC} ===")
    print(tests.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n{len(figures) + 1} figures and summary.csv written to {output}")
