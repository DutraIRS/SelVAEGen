import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from torch.nn import MSELoss

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from oracles import BaselineOracle, GraphDTAOracle, GSDTAOracle
from utils import RESULTS, save_figure, save_plot_data
from utils.data_tools import create_dataloaders
from utils.experiment_tools import provenance, save_config, setup_compute
from utils.stats_tools import qq_plot, sample_qq_plot

DATASETS = ["kiba", "bindingdb", "papyrus", "davis"]
REDRAW = False

SEED = 0
EPOCHS = 200
PATIENCE = 50
GRAD_CLIP = 10.0
BATCH_SIZE = 1024
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-10


def build_oracles(loader):
    """One of each oracle, sized from the batch it will consume."""
    batch = next(iter(loader))
    input_dim = batch.fingerprint.shape[1] + batch.prot_emb.shape[1]

    return [BaselineOracle(input_dim, 128, 3, 1), GraphDTAOracle(), GSDTAOracle()]

@torch.no_grad()
def predict(model, loader):
    """Predicted and observed affinity for every pair in a loader."""
    model.eval()
    predicted, observed = [], []

    for batch in loader:
        predicted.append(model(batch).view(-1).cpu())
        observed.append(batch.affinity.view(-1).cpu())

    return torch.cat(predicted), torch.cat(observed)

def selectivity_concordance(predicted, table, seed=0):
    """How often the oracle orders two targets of one drug the way the measurements do."""
    rng = np.random.default_rng(seed)
    affinity = table["affinity"].to_numpy()

    first, second = [], []
    for rows in table.groupby("drug_id").indices.values():
        if len(rows) > 1:
            shuffled = rng.permutation(rows)
            first.append(shuffled[:-1])
            second.append(shuffled[1:])

    if not first:
        return float("nan")                 # no drug has two measured targets

    first, second = np.concatenate(first), np.concatenate(second)
    measured = affinity[first] - affinity[second]
    inferred = predicted[first] - predicted[second]
    informative = measured != 0

    return float((np.sign(measured[informative]) == np.sign(inferred[informative])).mean())

def regression_metrics(predicted, observed):
    """The metrics the DTA literature reports, so the oracles can be placed against it."""
    predicted, observed = np.asarray(predicted), np.asarray(observed)

    # concordance index: how often the oracle orders a pair the way the measurements do
    order = np.argsort(observed)
    predicted, observed = predicted[order], observed[order]
    concordant = ties = pairs = 0.0
    for i in range(len(observed)):
        later = observed[i + 1:] > observed[i]
        if not later.any():
            continue
        gap = predicted[i + 1:][later] - predicted[i]
        pairs += gap.size
        concordant += float((gap > 0).sum())
        ties += float((gap == 0).sum())

    return {
        "ci": (concordant + 0.5 * ties) / pairs if pairs else float("nan"),
        "pearson": float(stats.pearsonr(predicted, observed).statistic),
        "spearman": float(stats.spearmanr(predicted, observed).statistic),
        "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
    }

def cluster_bootstrap_ci(values, clusters, statistic=np.mean, draws=2000, alpha=0.05, seed=0):
    """Percentile interval that resamples whole clusters, not individual interactions."""
    values = np.asarray(values, dtype=float)
    widest = None

    for name, key in clusters.items():
        key = np.asarray(key)
        groups = [np.flatnonzero(key == level) for level in np.unique(key)]
        rng = np.random.default_rng(seed)

        samples = []
        for _ in range(draws):
            picked = rng.integers(len(groups), size=len(groups))
            samples.append(statistic(values[np.concatenate([groups[g] for g in picked])]))

        low = float(np.quantile(samples, alpha / 2))
        high = float(np.quantile(samples, 1 - alpha / 2))

        if widest is None or high - low > widest[1] - widest[0]:
            widest = (low, high, name)

    return widest

def plot_history(history, name, directory):
    frame = pd.DataFrame(history).drop(columns=["seconds", "batches_skipped"], errors="ignore")
    frame = frame.melt("epoch", var_name="split", value_name="mse")
    save_plot_data(directory / f"{name}_loss.pdf", frame)

    return replot_history(frame, name, directory)

def replot_history(frame, name, directory):
    """The loss curve from the melted frame, which is also what its .csv holds."""
    axes = sns.lineplot(frame, x="epoch", y="mse", hue="split")
    axes.set_ylabel("MSE")

    return save_figure(axes.figure, directory / f"{name}_loss.pdf")

def train_concurrently(models, train_loader, val_loader, directory):
    """All oracles step on the same batches, one optimizer each; val decides what is saved."""
    named = []
    for model in models:
        name = model.__class__.__name__
        history = []
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        named.append((model, name, optimizer, history))

    criterion = MSELoss()
    best = {name: (float("inf"), 0) for _, name, _, _ in named}
    waiting = {name: 0 for _, name, _, _ in named}
    dropped = {name: 0 for _, name, _, _ in named}
    active = {name for _, name, _, _ in named}
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        for model, name, _, _ in named:
            if name in active:
                model.train()

        running = {name: 0.0 for _, name, _, _ in named}
        skipped = {name: 0 for _, name, _, _ in named}
        for batch in train_loader:
            for model, name, optimizer, _ in named:
                if name not in active:
                    continue

                loss = criterion(model(batch), batch.affinity.view(-1, 1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
                if not np.isfinite(norm):
                    optimizer.zero_grad(set_to_none=True)
                    skipped[name] += 1
                    dropped[name] += 1
                    continue

                optimizer.step()
                running[name] += loss.item()

        totals = {name: 0.0 for _, name, _, _ in named}
        for model, name, _, _ in named:
            if name in active:
                model.eval()

        validated = 0
        with torch.no_grad():
            for batch in val_loader:
                target = batch.affinity.view(-1, 1)
                validated += target.shape[0]
                for model, name, _, _ in named:
                    if name in active:
                        totals[name] += criterion(model(batch), target).item() * target.shape[0]

        for model, name, _, history in named:
            if name not in active:
                continue

            val_mse = totals[name] / max(validated, 1)
            history.append({"epoch": epoch, "train": running[name] / len(train_loader),
                            "val": val_mse, "seconds": time.perf_counter() - started,
                            "batches_skipped": skipped[name]})

            if val_mse < best[name][0]:
                best[name] = (val_mse, epoch)
                waiting[name] = 0
                torch.save(model, directory / f"{name}.pt")
            else:
                waiting[name] += 1

            print(f"{directory.name} | {name} | epoch {epoch:3d} | "
                    f"train {history[-1]['train']:.4f} | val {val_mse:.4f} | "
                    f"best {best[name][0]:.4f} @ {best[name][1]}"
                    + (f" | skipped {skipped[name]} this epoch, "
                        f"{dropped[name]} total" if dropped[name] else ""))

            if waiting[name] >= PATIENCE:
                active.discard(name)
                print(f"{directory.name} | {name} | retired at epoch {epoch}, "
                        f"best {best[name][0]:.4f} @ {best[name][1]}")

        if not active:
            print(f"{directory.name} | all oracles retired at epoch {epoch}")
            break

    return {name: (model, history, best[name]) for model, name, _, history in named}

def run_dataset(dataset, device):
    """The rows from training, scoring and plotting every oracle for one dataset."""
    records = []
    (train_loader, val_loader, test_loader), (_, _, test_table) = create_dataloaders(
        dataset, batch_size=BATCH_SIZE, train_frac=0.8, val_frac=0.1, test_frac=0.1,
        cold_protein=False, device=device)

    directory = RESULTS / dataset / "oracles"
    directory.mkdir(parents=True, exist_ok=True)
    save_config(directory, sys.modules[__name__], dataset=dataset)

    torch.manual_seed(SEED)

    trained = train_concurrently(
        [model.to(device) for model in build_oracles(train_loader)],
        train_loader, val_loader, directory)

    for name, (_, history, (best_mse, best_epoch)) in trained.items():
        plot_history(history, name, directory)

        model = torch.load(directory / f"{name}.pt", map_location=device, weights_only=False).eval()
        predicted, observed = predict(model, test_loader)
        test_mse = torch.nn.functional.mse_loss(predicted, observed).item()
        concordance = selectivity_concordance(predicted.numpy(), test_table)
        regression = regression_metrics(predicted.numpy(), observed.numpy())
        squared = (predicted - observed).numpy() ** 2
        mse_low, mse_high, mse_cluster = cluster_bootstrap_ci(
            squared, {"drug": test_table["drug_id"].to_numpy(), "protein": test_table["protein_id"].to_numpy()})

        print(f"{dataset} | {name} | test MSE {test_mse:.4f} "
                f"[{mse_low:.4f}, {mse_high:.4f}] ({mse_cluster}-clustered) | "
                f"CI {regression['ci']:.4f} | "
                f"pearson {regression['pearson']:.4f} | "
                f"spearman {regression['spearman']:.4f} | "
                f"selectivity concordance {concordance:.4f}", flush=True)

        qq_plot((predicted - observed).numpy(), path=directory / f"{name}_qq_residuals.pdf")
        sample_qq_plot(predicted.numpy(), observed.numpy(),
                sample_label="predicted affinity", reference_label="observed affinity",
                path=directory / f"{name}_qq_predicted.pdf")

        records.append({"dataset": dataset, "oracle": name, "test_mse": test_mse,
                        "test_mse_low": mse_low, "test_mse_high": mse_high,
                        "test_mse_ci_cluster": mse_cluster,
                        **regression,
                        "selectivity_concordance": concordance,
                        "best_val_mse": best_mse, "best_epoch": best_epoch,
                        "epoch": history[-1]["epoch"], "val": history[-1]["val"],
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "train_seconds": history[-1]["seconds"],
                        "seconds_per_epoch": history[-1]["seconds"] / len(history),
                        **provenance()})
        
        pd.DataFrame(records).to_csv(directory / "summary.csv", index=False)

    return records


def redraw_dataset(dataset, device):
    """Every oracle figure again, without retraining anything."""
    directory = RESULTS / dataset / "oracles"
    checkpoints = sorted(directory.glob("*.pt"))
    if not checkpoints:
        print(f"  {dataset}: no oracles on disk, nothing to redraw", flush=True)
        return 0

    (_, _, test_loader), (_, _, _) = create_dataloaders(
        dataset, batch_size=BATCH_SIZE, train_frac=0.8, val_frac=0.1, test_frac=0.1,
        cold_protein=False, device=device)

    drawn = 0
    for path in checkpoints:
        name = path.stem
        curve = directory / f"{name}_loss.csv"
        if curve.exists():
            replot_history(pd.read_csv(curve), name, directory)
            drawn += 1

        model = torch.load(path, map_location=device, weights_only=False).eval()
        predicted, observed = predict(model, test_loader)

        qq_plot((predicted - observed).numpy(), path=directory / f"{name}_qq_residuals.pdf")
        sample_qq_plot(predicted.numpy(), observed.numpy(),
                sample_label="predicted affinity", reference_label="observed affinity",
                path=directory / f"{name}_qq_predicted.pdf")
        drawn += 2
        print(f"  {dataset} | {name} | redrew", flush=True)

    return drawn


def main():
    device = setup_compute()
    records = []

    for dataset in DATASETS:
        print(f"\n=== {dataset} ===", flush=True)
        if REDRAW:
            redraw_dataset(dataset, device)
        else:
            records.extend(run_dataset(dataset, device))

    return pd.DataFrame(records)


if __name__ == "__main__":
    summary = main()
    
    if not summary.empty:
        print(summary.to_string(index=False))
