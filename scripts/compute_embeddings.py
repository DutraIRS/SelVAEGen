from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

CODE_DIR = Path(__file__).resolve().parents[1]
MAX_LENGTH = 2000
DATASETS = ["kiba", "bindingdb", "papyrus", "davis"]
LATENT_DIM = 128


def load_proteins(dataset):
    proteins = pd.read_csv(CODE_DIR / "data" / dataset / "proteins.csv", index_col=0).sort_index()

    return proteins["target_sequence"].tolist()

def embed_esmc(sequences, checkpoint, device, max_length):
    """ESMC's API takes one sequence at a time."""
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    model = ESMC.from_pretrained(checkpoint).to(device=device, dtype=torch.float32).eval()
    config = LogitsConfig(sequence=True, return_embeddings=True)

    output = []
    with torch.inference_mode():
        for index, sequence in enumerate(sequences, 1):
            encoded = model.encode(ESMProtein(sequence=sequence[:max_length]))
            states = model.logits(encoded, config).embeddings[0, 1:-1]
            output.append(states.float().mean(dim=0).cpu())

            if index % 100 == 0:
                print(f"    {index:,}/{len(sequences):,}")

    return torch.stack(output)

MODELS = {"esmc_600m": (embed_esmc, "esmc_600m", 1152)}

def reduce_embeddings(embeddings, latent_dim=LATENT_DIM):
    """Put a protein encoder's output into the geometry the prior N(m(P), I) assumes."""
    values = embeddings.numpy().astype(np.float64)

    mean, spread = values.mean(0), np.clip(values.std(0), 1e-6, None)
    values = (values - mean) / spread

    components = min(latent_dim, *values.shape)
    pca = PCA(n_components=components, random_state=0)
    values = pca.fit_transform(values)

    scale = float(values.std())
    values = values / scale

    fitted = {"pca": pca, "mean": mean, "std": spread, "scale": scale,
                "source_dim": int(embeddings.shape[1]), "latent_dim": components}

    return torch.tensor(values, dtype=torch.float32).contiguous(), fitted

def centre_spacing(embeddings):
    """Median distance between target centres, against the sqrt(2d) the prior implies."""
    distances = torch.cdist(embeddings, embeddings)
    distances = distances[~torch.eye(len(embeddings), dtype=torch.bool)]

    return float(distances.median()), (2 * embeddings.shape[1]) ** 0.5

def compute_and_save(dataset, model_name, device, max_length=MAX_LENGTH, latent_dim=LATENT_DIM):
    directory = CODE_DIR / "data" / dataset
    output_path = directory / f"protein_embs_{model_name}.pt"
    raw_path = directory / f"protein_embs_{model_name}_raw.pt"
    pca_path = directory / f"protein_emb_pca_{model_name}.joblib"

    embed, checkpoint, expected_dim = MODELS[model_name]
    sequences = load_proteins(dataset)

    truncated = sum(len(sequence) > max_length for sequence in sequences)
    print(f"  {model_name:<12} {len(sequences):,} proteins, {truncated:,} truncated at {max_length:,}")

    embeddings = embed(sequences, checkpoint, device, max_length).float().contiguous()

    expected_shape = (len(sequences), expected_dim)
    if embeddings.shape != expected_shape:
        raise ValueError(f"{model_name} produced {tuple(embeddings.shape)}, expected {expected_shape}")

    torch.save(embeddings, raw_path)

    reduced, fitted = reduce_embeddings(embeddings, latent_dim=latent_dim)
    torch.save(reduced, output_path)

    joblib.dump(fitted, pca_path)

    variance = float(fitted["pca"].explained_variance_ratio_.sum())
    spacing, expected = centre_spacing(reduced)

    size_mib = output_path.stat().st_size / 1024**2
    print(f"  {'':12} {expected_dim} -> {fitted['latent_dim']} dims, {variance:.1%} of variance kept")
    print(f"  {'':12} median centre spacing {spacing:.1f} vs sqrt(2d) = {expected:.1f} ({spacing / expected:.0%})")
    print(f"  {'':12} wrote {output_path.name}  {tuple(reduced.shape)}  {size_mib:.1f} MiB (+ {raw_path.name}, {pca_path.name})")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}  |  datasets {', '.join(DATASETS)}")

    for dataset in DATASETS:
        print(dataset)
        for model_name in MODELS:
            compute_and_save(dataset, model_name, device)

    print("done!")

if __name__ == "__main__":
    main()
