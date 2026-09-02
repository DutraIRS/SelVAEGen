import os
import sys
from pathlib import Path

import torch
import joblib
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from rdkit import Chem
from sklearn.decomposition import PCA

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.chemical_tools import compute_fingerprints
from utils.data_tools import MAX_SMILES_LEN, load_tokenizer

DATASETS = ["kiba", "bindingdb", "papyrus", "davis"]
FP_DIM = 2048         # bits per hashed fingerprint, before PCA
FINAL_DIM = 60        # components kept
JOBS = os.cpu_count()

def screen_smiles(smiles, tokenizer):
    """Same validity rules as the dataloader: short enough to tokenize"""
    if tokenizer.encode(smiles, max_length=MAX_SMILES_LEN) is None:
        return False
    if "." in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    return len(Chem.GetMolFrags(mol)) <= 1

def process_dataset(dataset, fp_dim=512, final_dim=512, n_jobs=8):
    drugs = pd.read_csv(CODE_DIR / "data" / dataset / "drugs.csv", index_col=0).sort_index()
    tokenizer = load_tokenizer(dataset)

    keep_mask = [screen_smiles(smi, tokenizer) for smi in drugs["smiles"]]
    kept = drugs[keep_mask]
    print(f"{dataset}: {len(kept):,}/{len(drugs):,} drugs pass the single-fragment screen")

    fp_tensors = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(compute_fingerprints)(smi, fp_dim=fp_dim, device="cpu")
        for smi in kept["smiles"]
    )

    fps = torch.stack(fp_tensors, dim=0).numpy()

    print(f"Computed fingerprints for {len(fps)} drugs in {dataset} dataset ({fps.shape}).")

    pca = PCA(n_components=final_dim)
    reduced_fps = pca.fit_transform(fps)

    print(
        f"Maintained variance: "
        f"{np.sum(pca.explained_variance_ratio_):.4f} "
        f"after PCA reduction to {final_dim} dimensions."
    )

    torch.save(torch.tensor(reduced_fps, dtype=torch.float32), CODE_DIR / "data" / dataset / "drug_fingerprints.pt")

    # which drug_id each fingerprint row belongs to, for downstream alignment
    torch.save(torch.tensor(kept.index.to_numpy(), dtype=torch.long), CODE_DIR / "data" / dataset / "drug_fp_ids.pt")

    # save the fitted basis, so new molecules get the training transformation
    joblib.dump({"pca": pca, "fp_dim": fp_dim}, CODE_DIR / "data" / dataset / "drug_fingerprint_pca.joblib")


def main():
    for dataset in DATASETS:
        print(f"\n=== {dataset}", flush=True)
        process_dataset(dataset, fp_dim=FP_DIM, final_dim=FINAL_DIM, n_jobs=JOBS)


if __name__ == "__main__":
    main()
