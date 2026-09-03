# SelVAEGen

Selective VAE Generator.
First presented as dissertation for the MSc in Applied Mathematics and Data Science at Getulio Vargas Foundation (FGV) by Iago Dutra.

## Structure
The repository follows this organization:

```
data/<dataset>/        affinities.csv, drugs.csv, proteins.csv, and the tensors the scripts produce
docs/                  the dissertation's approved version and the presentation slides
experiments/           where action happens
models/                SelVAEGen and competitors
oracles/               the three affinity predictors
scripts/               one-time data preparation
utils/                 losses, metrics, statistics, data and experiment plumbing
```

## Environment

In the following, Ubuntu 22.04.4 LTS, Python 3.11.0, and CUDA 12.4 are assumed. Minor adaptations may be required according to available GPU.

### Setting up an environment
To create a virtual environment and install dependencies, run the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Computing fingerprints and protein data
To pre-compute drug fingerprints, protein embeddings, names and similarity matrices, run the following scripts:

```bash
python scripts/compute_fingerprints.py
python scripts/compute_embeddings.py
python scripts/run_blast.py
python scripts/fetch_protein_names.py
```

## Running experiments
Experiment `e1` must be run for every dataset before anything else. After that, experiments `e2`, `e3`, and `e4` can be run in parallel.
With their results, `e5` can be run on the chosen architecture. Finally, `e6` trains the models evaluated in `e7`, `e8` and `e9`.
In all cases, you can just run:
```bash
python experiments/<experiment_name>.py
```
