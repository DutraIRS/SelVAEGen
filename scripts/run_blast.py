import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils import DATA

DATASETS = ["kiba", "davis", "papyrus", "bindingdb"]

# blastp's defaults for protein search (blastp)
MATRIX = "BLOSUM62"
OPEN_GAP = -11.0
EXTEND_GAP = -1.0

WORKERS = None # None lets Pool take the machine's cpu count

_SEQUENCES = []
_ALIGNER = None


def aligner():
    """A local aligner configured the way blastp scores protein pairs."""
    engine = Align.PairwiseAligner(scoring=None)
    engine.mode = "local"
    engine.substitution_matrix = substitution_matrices.load(MATRIX)
    engine.open_gap_score = OPEN_GAP
    engine.extend_gap_score = EXTEND_GAP

    return engine

def clean(sequence, alphabet):
    """Uppercase, with anything the matrix has no column for read as unknown."""
    return "".join(residue if residue in alphabet else "X" for residue in sequence.upper())

def percent_identity(alignment):
    """Identical columns over the aligned length, as blastp reports it."""
    target, query = alignment[0], alignment[1]
    matches = sum(a == b for a, b in zip(target, query) if a != "-" and b != "-")
    columns = sum(1 for a, b in zip(target, query) if a != "-" or b != "-")

    return 100.0 * matches / columns

def _start_worker(sequences):
    """Each process keeps its own aligner and its own copy of the sequences."""
    global _SEQUENCES, _ALIGNER
    _SEQUENCES = sequences
    _ALIGNER = aligner()

def _identity_row(index):
    """Row `index` of the upper triangle, against every later sequence."""
    row = np.zeros(len(_SEQUENCES), dtype=np.float32)
    for other in range(index + 1, len(_SEQUENCES)):
        row[other] = percent_identity(_ALIGNER.align(_SEQUENCES[index], _SEQUENCES[other])[0])

    return index, row

def identity_matrix(sequences):
    """Symmetric n x n percent identity, one process per upper-triangle row."""
    size = len(sequences)
    matrix = np.eye(size, dtype=np.float32) * 100.0

    with Pool(WORKERS, initializer=_start_worker, initargs=(sequences,)) as pool:
        for done, (index, row) in enumerate(pool.imap_unordered(_identity_row, range(size), chunksize=4), 1):
            matrix[index, index + 1:] = row[index + 1:]
            matrix[index + 1:, index] = row[index + 1:]
            if done % 100 == 0:
                print(f"    {done:,}/{size:,} rows", flush=True)

    return matrix

def run_dataset(dataset):
    """Write the protein identity matrix and its ids for one dataset."""
    path = DATA / dataset / "protein_identity.npz"
    if path.exists():
        print(f"{dataset}: {path.name} already written, skipped", flush=True)
        return

    proteins = pd.read_csv(DATA / dataset / "proteins.csv", index_col=0).sort_index()
    alphabet = set(substitution_matrices.load(MATRIX).alphabet)
    sequences = [clean(sequence, alphabet) for sequence in proteins["target_sequence"]]

    print(f"{dataset}: {len(sequences):,} proteins, {len(sequences) * (len(sequences) - 1) // 2:,} pairs", flush=True)
    matrix = identity_matrix(sequences)

    np.savez_compressed(path, identity=matrix, protein_ids=np.asarray(proteins.index, dtype=np.int64))
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    print(f"  wrote {path.name}: median {np.median(off_diagonal):.1f}%, max {off_diagonal.max():.1f}%", flush=True)


def main():
    for dataset in DATASETS:
        run_dataset(dataset)


if __name__ == "__main__":
    main()
