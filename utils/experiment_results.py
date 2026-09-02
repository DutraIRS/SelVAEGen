import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import RESULTS

LIBRARY_METRICS = ("validity", "strict_validity", "bonds_skipped", "atoms_dropped",
                    "uniqueness", "novelty", "yield", "internal_diversity",
                    "mean_atoms", "reference_mean_atoms", "mean_atoms_ratio", "qed",
                    "synthetic_accessibility", "ro5", "ro5_conditions", "tanimoto_gap")

ORACLE_METRICS = ("average_pba", "average_top10", "success_rate", "delta_score",
                    "distributional_delta_score", "selectivity_effect",
                    "selectivity_spread", "directional_consistency", "hit_delta_score",
                    "effective_delta", "effective_count", "hit_count",
                    "oracle_disagreement", "scored_count")

ORACLE_NAMES = ("GraphDTAOracle", "GSDTAOracle", "BaselineOracle")

PRIMARY_METRICS = ("effective_delta", "directional_consistency", "tanimoto_gap")

DECISION_METRIC = "effective_delta"

BRANCH_AGGREGATION = "mean"

SUCCESS_THRESHOLD = {"kiba": 12.1, "bindingdb": 7.0, "papyrus": 7.0, "davis": 7.0}

def env_seeds(variable, default):
    spec = os.environ.get(variable, "").strip()
    if not spec:
        return default

    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            low, _, high = part.partition("-")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))

    return seeds


def decision_column(frame, branch, metric, oracle="MeanOracle"):
    """The column holding this metric, whichever shape the experiment wrote."""
    for candidate in (f"{branch}_{metric}", f"{branch}_{oracle}_{metric}"):
        if candidate in frame and frame[candidate].notna().any():
            return candidate

    return None

def branch_decision(frame, metric=DECISION_METRIC, oracle="MeanOracle",
                    branches=("seq", "graph"), how=BRANCH_AGGREGATION):
    """One number per row: `metric` aggregated over the decoder branches that arm has."""
    columns = [c for c in (decision_column(frame, b, metric, oracle) for b in branches) if c is not None]

    if not columns:
        return pd.Series(float("nan"), index=frame.index)

    return getattr(frame[columns], how)(axis=1)

def metric_family(column, primary_oracle="MeanOracle"):
    """"primary" for a confirmatory column, "exploratory" for the rest."""
    body = column.split("_", 1)[-1]                    # drop the seq_/graph_ prefix

    if any(body in (name, f"{primary_oracle}_{name}") for name in PRIMARY_METRICS):
        return "primary"

    return "exploratory"

def experiment_dir(dataset, experiment):
    """results/<dataset>/<experiment>/."""
    path = RESULTS / dataset / experiment
    path.mkdir(parents=True, exist_ok=True)

    return path

def run_dir(dataset, experiment, name):
    """results/<dataset>/<experiment>/runs/<name>/."""
    path = experiment_dir(dataset, experiment) / "runs" / name
    path.mkdir(parents=True, exist_ok=True)

    return path

def save_config(directory, module, **extra):
    """Every constant the experiment ran with, written beside its results.csv."""
    config = {name: value for name, value in vars(module).items() if name.isupper()}
    config = {**config, **extra, **provenance()}

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2, default=str),
                                            encoding="utf-8")

    return config

def _jsonable(value):
    """numpy scalars as the Python scalars they are, so a seed stays an int on reload."""
    if isinstance(value, np.generic):
        return value.item()

    return str(value)

def _key_part(value):
    """Canonical form of one key field."""
    if isinstance(value, (bool, str)):
        return str(value)

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if np.isnan(number):                # NaN is never a usable key
        return "nan"

    return str(int(number)) if number.is_integer() else repr(number)

def record_run(directory, row):
    """One finished run's result row, written beside that run's own artifacts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / "row.json"
    staging = directory / f"row.json.{os.getpid()}.tmp"
    staging.write_text(json.dumps(row, default=_jsonable), encoding="utf-8")
    os.replace(staging, path)          # atomic

    return path

def collect_runs(output, keys, write=True, valid=None):
    """(rows finished by any job, is-this-one-finished), and refresh results.csv."""
    output = Path(output)
    rows, seen, foreign = [], set(), set()

    allowed = ({key: {_key_part(v) for v in values} for key, values in valid.items()} if valid else {})

    def take(row):
        if not set(keys) <= set(row):
            return                     # a file that predates these keys skips nothing

        identity = tuple(_key_part(row[k]) for k in keys)
        if identity in seen:
            # a run reaches here twice, once from its row.json and once from results.csv
            return

        if any(_key_part(row[key]) not in values for key, values in allowed.items()
                if key in row):
            foreign.add(identity)

        seen.add(identity)
        rows.append(row)

    found = sorted((output / "runs").glob("*/row.json"))
    for path in found:
        try:
            take(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue                   # a job is mid-write; the next scan will catch it

    legacy = output / "results.csv"
    if legacy.exists():
        try:
            frame = pd.read_csv(legacy)
            if not frame.empty:
                for row in frame.to_dict("records"):
                    take(row)
        except (pd.errors.EmptyDataError, OSError):
            pass

    if foreign:
        print(f"  {len(foreign)} recorded runs fall outside this instance's grid (a sibling job's seeds) and are kept in results.csv")

    if rows:
        print(f"  {len(rows)} runs already recorded under {output.name}")
        if write:
            staging = output / f"results.csv.{os.getpid()}.tmp"
            pd.DataFrame(rows).to_csv(staging, index=False)
            os.replace(staging, legacy)

    return rows, (lambda **kw: tuple(_key_part(kw[k]) for k in keys) in seen)

def complete_blocks(frame, arm_column, block_column="seed", warn=True):
    """Only the blocks that ran every arm, so a seed counts the same way everywhere."""
    if frame.empty or arm_column not in frame or block_column not in frame:
        return frame

    arms = frame[arm_column].dropna().nunique()
    per_block = frame.groupby(block_column)[arm_column].nunique()
    keep = per_block[per_block == arms].index

    if warn and len(keep) < len(per_block):
        missing = {block: sorted(set(frame[arm_column].dropna().unique())
                                    - set(frame.loc[frame[block_column] == block, arm_column]))
                    for block in per_block.index.difference(keep)}
        for block, absent in missing.items():
            print(f"  [complete_blocks] {block_column} {block} dropped: never ran {absent}")

    return frame[frame[block_column].isin(keep)]

def grid_complete(finished, combinations, label=""):
    """True when every (key -> value) combination in the grid has already been recorded."""
    missing = [combo for combo in combinations if not finished(**combo)]
    if missing:
        print(f"  {len(missing)} of {len(combinations)} {label or 'runs'} still to do")
        return False

    print(f"  all {len(combinations)} {label or 'runs'} already present -- nothing to train")
    return True

def provenance():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10, check=False,
                                cwd=str(RESULTS.parent)).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"

    device = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    return {"commit": commit, "torch": torch.__version__, "device": device}
