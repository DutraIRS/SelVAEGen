import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

import torch

from torch.utils.data import Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from . import DATA
from .chemical_tools import compute_fingerprints, make_graph, safe_smiles
from .tokenizer import SmilesTokenizer, SelfiesTokenizer


MAX_SEQ_LEN = 2000
SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: i + 1 for i, v in enumerate(SEQ_VOC)}
MAX_SMILES_LEN = 128
MIN_PROTEIN_LEN = 50
MAX_PROTEIN_LEN = 2000
MIN_MEASUREMENTS = 5
MAX_AFFINITY_DEVIATIONS = 12.0
PROT_EMB_MODEL = os.environ.get("PROT_EMB_MODEL", "esmc_600m")


def encode_sequence(sequence, max_len=MAX_SEQ_LEN, device="cpu"):
    """Label-encode a protein: 1-25 for residues, 0 for padding."""
    sequence = sequence.upper()

    x = torch.zeros(max_len, dtype=torch.long, device=device)
    for i, ch in enumerate(sequence[:max_len]):
        if ch not in SEQ_DICT:
            raise ValueError(f"residue {ch!r} at position {i} is not in {SEQ_VOC}")
        x[i] = SEQ_DICT[ch]

    return x.unsqueeze(0)

def drop_affinity_outliers(affinities, deviations=MAX_AFFINITY_DEVIATIONS):
    """Drop rows whose affinity sits more than `deviations` scaled MADs from the median."""
    if deviations is None or affinities.empty:
        return affinities

    values = affinities["affinity"].to_numpy()
    median = np.median(values)
    spread = np.median(np.abs(values - median)) * 1.4826

    if not spread:
        return affinities

    keep = np.abs(values - median) <= deviations * spread
    if not keep.all():
        dropped = np.sort(values[~keep])
        print(f"dropped {(~keep).sum():,} affinities outside "
                f"[{median - deviations * spread:.2f}, {median + deviations * spread:.2f}]: "
                f"{np.round(dropped[:5], 3).tolist()}{' ...' if (~keep).sum() > 5 else ''}")

    return affinities[keep]

def make_datapoints(drugs, affinities):
    graph_cache = {}
    data = []
    for _, row in affinities.iterrows():
        drug_inx = int(row["drug_id"])
        prot_inx = int(row["protein_id"])
        affinity = float(row["affinity"])

        drug_smiles = drugs.loc[drug_inx, "smiles"]

        if drug_inx not in graph_cache:
            x, edge_index, edge_attr = make_graph(drug_smiles)
            graph_cache[drug_inx] = (x.cpu(), edge_index.cpu(), edge_attr.cpu())

        x, edge_index, edge_attr = graph_cache[drug_inx]

        datapoint = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=drug_smiles,
            drug_id=torch.tensor([drug_inx], dtype=torch.long),
            protein_id=torch.tensor([prot_inx], dtype=torch.long),
            affinity=torch.tensor([affinity], dtype=torch.float32),
        )
        data.append(datapoint)

    return data

def make_split(affinities, train_frac=0.8, val_frac=0.1, test_frac=0.1, cold_protein=True, random_state=42):
    """Whole proteins are held out together when cold_protein."""
    fractions = np.array([train_frac, val_frac, test_frac], dtype=float)
    fractions /= fractions.sum()

    proteins = affinities["protein_id"]
    keys = proteins.unique() if cold_protein else affinities.index.to_numpy()
    keys = np.random.default_rng(random_state).permutation(keys)

    cuts = (np.cumsum(fractions)[:2] * len(keys)).astype(int)

    splits = []
    for group in np.split(keys, cuts):
        selected = proteins.isin(group) if cold_protein else affinities.index.isin(group)
        splits.append(affinities[selected].copy())

    return tuple(splits)

def build_protein_table(proteins, encode_len=MAX_SEQ_LEN):
    """Protein sequence table on CPU, row i being protein_id i."""
    return torch.cat([
        encode_sequence(sequence, max_len=encode_len, device="cpu")
        for sequence in proteins["target_sequence"]
    ], dim=0)

def build_drug_token_table(drugs, tokenizer, max_smiles_len=MAX_SMILES_LEN):
    """Build fixed-length drug token table on CPU aligned by drug_id."""
    token_rows = []
    keep_drug_ids = set()
    pad_row = torch.full((max_smiles_len,), tokenizer.pad_id, dtype=torch.long)

    for drug_id, smiles in drugs["smiles"].items():
        tokens = tokenizer.encode(smiles, max_length=max_smiles_len, device="cpu")
        if tokens is None:
            token_rows.append(pad_row.clone())
            continue

        keep_drug_ids.add(drug_id)
        token_rows.append(tokens)

    return torch.stack(token_rows, dim=0), keep_drug_ids

def attach_features(batch, loader):
    """Attach the shared molecule and protein tables to a collated batch."""
    drug_ids = batch.drug_id.view(-1).cpu()
    protein_ids = batch.protein_id.view(-1).cpu()

    if loader.with_protein_sequence:
        batch.prot_seq_cat = loader.protein_table[protein_ids]

    batch.prot_emb = loader.protein_embs[protein_ids]
    batch.fingerprint = loader.fingerprint_table[drug_ids]
    batch.drug_tokens = loader.drug_token_table[drug_ids]

    return batch

class ProteinGroupedSampler(Sampler):
    """Batch row indices drawn from a few targets at a time."""
    def __init__(self, protein_ids, batch_size, groups_per_batch=4, seed=0):
        self.groups = [np.flatnonzero(protein_ids == protein) for protein in np.unique(protein_ids)]
        self.groups_per_batch = groups_per_batch
        self.rows_per_group = batch_size // groups_per_batch
        # a partial last batch is dropped, so every batch holds the same target count
        self.num_batches = len(protein_ids) // batch_size

        if self.num_batches == 0:
            raise ValueError(f"batch_size {batch_size} exceeds the {len(protein_ids)} rows in this split, so an epoch would take no steps")
        if len(self.groups) < groups_per_batch:
            raise ValueError(f"this split has {len(self.groups)} proteins but groups_per_batch is {groups_per_batch}")
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        for _ in range(self.num_batches):
            chosen = rng.choice(len(self.groups), self.groups_per_batch, replace=False)
            yield np.concatenate([
                rng.choice(self.groups[group], self.rows_per_group, replace=len(self.groups[group]) < self.rows_per_group)
                for group in chosen]).tolist()

    def __len__(self):
        return self.num_batches

class FeaturesLoader:
    """Attaches the shared molecule and protein features to each batch, then moves it."""
    def __init__(self, loader, protein_table, protein_embs,
                fingerprint_table, drug_token_table, device='cpu', with_protein_sequence=True):
        self.loader = loader
        self.with_protein_sequence = with_protein_sequence
        self.protein_table = protein_table.cpu()
        self.protein_embs = protein_embs.cpu()
        self.fingerprint_table = fingerprint_table.cpu()
        self.drug_token_table = drug_token_table.cpu()
        self.device = device

    def __iter__(self):
        for batch in self.loader:
            yield attach_features(batch, self).to(self.device)

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset


def reset_loader_epoch(loader, epoch=0):
    """Rewind a ProteinGroupedSampler to `epoch`."""
    sampler = getattr(getattr(loader, "loader", loader), "batch_sampler", None)

    if hasattr(sampler, "epoch"):
        sampler.epoch = int(epoch)

    return loader

def build_generated_loader(mols, protein_ids, protein_table, protein_embs, tokenizer,
                            fingerprint_pca=None, max_smiles_len=MAX_SMILES_LEN,
                            batch_size=256, device="cpu"):
    """Cross generated molecules with a panel of targets, for one batched oracle pass."""
    from rdkit import Chem

    if fingerprint_pca is None:
        raise ValueError("fingerprint_pca is required: generated molecules have to go through the same basis the training rows went through")

    pad_row = torch.full((max_smiles_len,), tokenizer.pad_id, dtype=torch.long)

    smiles, tokens = [], []
    for mol in mols:
        if mol is None:
            continue

        text = safe_smiles(mol)
        if text is None or Chem.MolFromSmiles(text) is None:
            continue

        row = tokenizer.encode(text, max_length=max_smiles_len, device="cpu")

        smiles.append(text)
        tokens.append(pad_row.clone() if row is None else row)

    if not smiles:
        return None, []

    pairs = pd.DataFrame([(drug_id, protein_id, 0.0) for drug_id in range(len(smiles)) for protein_id in protein_ids],
                        columns=["drug_id", "protein_id", "affinity"])
    data = make_datapoints(pd.DataFrame({"smiles": smiles}), pairs)

    raw = torch.stack([compute_fingerprints(smi, fp_dim=fingerprint_pca["fp_dim"]) for smi in smiles])
    fingerprints = torch.tensor(fingerprint_pca["pca"].transform(raw.numpy()), dtype=torch.float32)

    # score_with sets prot_seq_cat itself, one target at a time
    loader = FeaturesLoader(
        DataLoader(data, batch_size=batch_size, shuffle=False),
        protein_table, protein_embs,
        fingerprints, torch.stack(tokens), device=device,
        with_protein_sequence=False)

    return loader, smiles

def shared_drug_ids(dataset, tokenizers, max_smiles_len=MAX_SMILES_LEN):
    """drug_ids every tokenizer can encode within the length cap."""
    drugs = pd.read_csv(DATA / dataset / "drugs.csv", index_col=0)

    keep = None
    for tokenizer in tokenizers:
        encodable = {drug_id for drug_id, smiles in drugs["smiles"].items() if tokenizer.encode(smiles, max_length=max_smiles_len) is not None}
        keep = encodable if keep is None else keep & encodable

    return keep

@lru_cache(maxsize=None)
def load_tokenizer(dataset, use_selfies=False):
    """Fit the shared SMILES or SELFIES tokenizer on the full drug vocabulary."""
    drugs = pd.read_csv(DATA / dataset / "drugs.csv", index_col=0)

    if not use_selfies:
        return SmilesTokenizer.fit(drugs["smiles"])

    cached = DATA / dataset / "selfies_alphabet.json"
    if cached.exists():
        return SelfiesTokenizer(extra=json.loads(cached.read_text(encoding="utf-8")))

    tokenizer = SelfiesTokenizer.fit(drugs["smiles"])
    cached.write_text(json.dumps(tokenizer.extra, indent=2), encoding="utf-8")

    return tokenizer

def protein_geometry(embeddings, sample=256, seed=0):
    """Median centre spacing against the sqrt(2d) the unit-variance prior implies."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(embeddings), min(sample, len(embeddings)), replace=False)
    picked = embeddings[torch.as_tensor(rows)]

    distances = torch.cdist(picked, picked)
    distances = distances[~torch.eye(len(rows), dtype=torch.bool)]

    expected = (2 * embeddings.shape[1]) ** 0.5

    return float(distances.median()), expected

def load_protein_embeddings(dataset, prot_emb_model=None):
    """Protein embeddings as saved, row i being protein_id i, already at the latent width."""
    embs = torch.load(DATA / dataset / f"protein_embs_{prot_emb_model}.pt", map_location="cpu")

    return embs.float().contiguous()

def load_drug_fingerprint(dataset, n_drugs=None):
    """Precomputed drug fingerprints, row i being drug_id i in drugs.csv."""
    path = DATA / dataset / "drug_fingerprints.pt"
    ids_path = DATA / dataset / "drug_fp_ids.pt"

    table = torch.load(path, map_location="cpu").float()

    if not ids_path.exists():
        # without the index the only alignment that can be checked is one row per drug
        if n_drugs is not None and len(table) != n_drugs:
            raise ValueError(
                f"{path.name} holds {len(table):,} rows for {n_drugs:,} drugs and "
                f"{ids_path.name} is missing, so no row can be matched to a drug_id. "
                f"Rerun scripts/compute_fingerprints.py for {dataset}.")

        return table.contiguous()

    ids = torch.load(ids_path, map_location="cpu")

    if len(ids) != len(table):
        raise ValueError(f"{ids_path.name} indexes {len(ids):,} rows but {path.name} holds {len(table):,}")

    width = n_drugs if n_drugs is not None else int(ids.max()) + 1
    if int(ids.max()) >= width:
        raise ValueError(f"{ids_path.name} names drug_id {int(ids.max()):,}, outside the {width:,} drugs in {dataset}")

    full = torch.zeros(width, table.shape[1])
    full[ids] = table

    return full.contiguous()

def load_drug_fingerprint_ids(dataset):
    """Which drug_ids drug_fingerprints.pt actually holds, or None if it is unindexed."""
    path = DATA / dataset / "drug_fp_ids.pt"

    if not path.exists():
        return None

    return set(torch.load(path, map_location="cpu").tolist())

def load_fingerprint_pca(dataset):
    """The fitted basis and fp_dim behind a dataset's drug_fingerprints.pt."""
    import joblib

    return joblib.load(DATA / dataset / "drug_fingerprint_pca.joblib")

def create_dataloaders(dataset, tokenizer=None, use_selfies=False, batch_size=512,
                    train_frac=0.8, val_frac=0.1, test_frac=0.0, prot_emb_model=None,
                    max_smiles_len=MAX_SMILES_LEN, min_protein_len=MIN_PROTEIN_LEN,
                    max_protein_len=MAX_PROTEIN_LEN, min_measurements=MIN_MEASUREMENTS,
                    protein_encode_len=MAX_SEQ_LEN, random_state=42, device="cpu",
                    restrict_drug_ids=None, cold_protein=True, groups_per_batch=None,
                    with_protein_sequence=True, drop_multi_fragment=True,
                    max_affinity_deviations=MAX_AFFINITY_DEVIATIONS):
    prot_emb_model = prot_emb_model or PROT_EMB_MODEL

    drugs = pd.read_csv(DATA / dataset / "drugs.csv", index_col=0)
    proteins = pd.read_csv(DATA / dataset / "proteins.csv", index_col=0)
    affinities = pd.read_csv(DATA / dataset / "affinities.csv")

    print(
        f"Before filters:\n{dataset}: {len(affinities):,} interactions, "
        f"{affinities['drug_id'].nunique():,} drugs, "
        f"{affinities['protein_id'].nunique():,} proteins"
    )

    affinities = drop_affinity_outliers(affinities, max_affinity_deviations)

    if tokenizer is None:
        tokenizer = load_tokenizer(dataset, use_selfies=use_selfies)

    drug_token_table, keep_drug_ids = build_drug_token_table(drugs, tokenizer, max_smiles_len=max_smiles_len)
    protein_embs = load_protein_embeddings(dataset, prot_emb_model=prot_emb_model)

    spacing, expected = protein_geometry(protein_embs)
    print(f"protein embeddings: {prot_emb_model} -> {tuple(protein_embs.shape)} | "
            f"median centre spacing {spacing:.1f} vs sqrt(2d) = {expected:.1f} "
            f"({spacing / expected:.0%})")

    if drop_multi_fragment:
        salts = set(drugs.index[drugs["smiles"].str.contains(".", regex=False)])
        keep_drug_ids -= salts
        print(f"dropped {len(salts):,} multi-fragment drugs")

    fingerprinted = load_drug_fingerprint_ids(dataset)
    if fingerprinted is not None:
        missing = keep_drug_ids - fingerprinted
        if missing:
            print(f"dropped {len(missing):,} drugs with no fingerprint row")
            keep_drug_ids -= missing

    if restrict_drug_ids is not None:
        keep_drug_ids &= set(restrict_drug_ids)

    affinities = affinities[affinities["drug_id"].isin(keep_drug_ids)]

    protein_lengths = proteins["target_sequence"].str.len()
    keep_protein_ids = proteins.index[
        (protein_lengths >= min_protein_len) & (protein_lengths <= max_protein_len)
    ]
    affinities = affinities[affinities["protein_id"].isin(keep_protein_ids)]

    counts = affinities.groupby("protein_id")["protein_id"].transform("size")
    affinities = affinities[counts >= min_measurements]

    affinities = affinities.reset_index(drop=True)

    print(
        f"After filters:\n{dataset}: {len(affinities):,} interactions, "
        f"{affinities['drug_id'].nunique():,} drugs, "
        f"{affinities['protein_id'].nunique():,} proteins"
    )

    splits = make_split(affinities, train_frac=train_frac, val_frac=val_frac,
                        test_frac=test_frac, cold_protein=cold_protein,
                        random_state=random_state)

    train, val, test = splits

    print(
        f"split: {len(train):,} train / {len(val):,} val / {len(test):,} test | "
        f"{train['protein_id'].nunique():,} / {val['protein_id'].nunique():,} / "
        f"{test['protein_id'].nunique():,} proteins"
    )

    protein_table = build_protein_table(proteins, encode_len=protein_encode_len)
    fingerprint_table = load_drug_fingerprint(dataset, n_drugs=len(drugs))

    loaders = []
    for affs, shuffle in zip(splits, (True, False, False)):
        data = make_datapoints(drugs, affs)

        if shuffle and groups_per_batch:
            sampler = ProteinGroupedSampler(affs["protein_id"].to_numpy(), batch_size, groups_per_batch, seed=random_state)
            loader = DataLoader(data, batch_sampler=sampler)
        else:
            loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)

        loaders.append(FeaturesLoader(loader, protein_table, protein_embs,
                            fingerprint_table, drug_token_table, device=device,
                            with_protein_sequence=with_protein_sequence))

    return tuple(loaders), splits
