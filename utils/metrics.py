import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer
from rdkit.DataStructs import BulkTanimotoSimilarity

from .chemical_tools import safe_smiles


SPREAD_FLOOR = 0.1
LOSS_SPREAD_FLOOR = 1e-3


class Metrics:
    def __init__(self):
        from fcd import load_ref_model

        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.ref_model = load_ref_model()

    @staticmethod
    def _target_partition(affinities, target_idx):
        on_target = affinities[:, target_idx]
        off_target = torch.cat([affinities[:, :target_idx], affinities[:, target_idx + 1:]], dim=1)

        return on_target, off_target

    @staticmethod
    def _valid_mols(mols):
        return [mol for mol in mols
                if mol is not None and mol.GetNumAtoms() > 0
                and safe_smiles(mol) is not None]

    @staticmethod
    def _per_molecule(mols, measure):
        """`measure` over the molecules it can handle, skipping the ones RDKit refuses."""
        values = []
        for mol in mols:
            try:
                values.append(measure(mol))
            except Exception:                 # RDKit raises several unrelated C++ types
                continue

        return values

    def validity(self, mols):
        valid_mols = self._valid_mols(mols)
        score = len(valid_mols) / len(mols) if mols else float("nan")
        return score, valid_mols

    def uniqueness(self, mols):
        unique = {}

        for mol in self._valid_mols(mols):
            smiles = Chem.MolToSmiles(mol, canonical=True)
            unique.setdefault(smiles, mol)

        unique_mols = list(unique.values())
        score = len(unique_mols) / len(mols) if mols else float("nan")

        return score, unique_mols

    def novelty(self, mols, train_mols, known=None):
        """Fraction not already in the training set."""
        if known is None:
            known = {
                Chem.MolToSmiles(mol, canonical=True)
                for mol in self._valid_mols(train_mols)
            }

        novel = [
            mol for mol in self._valid_mols(mols)
            if Chem.MolToSmiles(mol, canonical=True) not in known
        ]

        return len(novel) / len(mols) if mols else float("nan")

    def internal_diversity(self, mols):
        fingerprints = self._per_molecule(self._valid_mols(mols), self.mfpgen.GetFingerprint)

        if len(fingerprints) < 2:
            return float("nan")

        similarities = []
        for i, fp in enumerate(fingerprints):
            similarities.extend(BulkTanimotoSimilarity(fp, fingerprints[i + 1:]))

        return 1.0 - float(np.mean(similarities))

    def qed(self, mols):
        scores = self._per_molecule(self._valid_mols(mols), QED.qed)
        return float(np.mean(scores)) if scores else float("nan")

    def synthetic_accessibility(self, mols):
        scores = self._per_molecule(self._valid_mols(mols), sascorer.calculateScore)
        return float(np.mean(scores)) if scores else float("nan")

    RULE_OF_FIVE = (
        ("mw", lambda mol: Descriptors.MolWt(mol) <= 500),
        ("logp", lambda mol: Crippen.MolLogP(mol) <= 5),
        ("hbd", lambda mol: rdMolDescriptors.CalcNumHBD(mol) <= 5),
        ("hba", lambda mol: rdMolDescriptors.CalcNumHBA(mol) <= 10),
    )
    ASSEMBLY_COMPROMISES = ("charges_dropped", "aromatics_demoted",
                            "bonds_skipped", "atoms_dropped")

    def strict_validity(self, mols):
        """Share of molecules the assembler built without altering the prediction at all."""
        built = [mol for mol in self._valid_mols(mols) if mol.HasProp("assembly_tier")]

        if not built:
            return float("nan")

        return float(np.mean([
            all(mol.GetIntProp(name) == 0 for name in self.ASSEMBLY_COMPROMISES)
            for mol in built]))

    def assembly_compromises(self, mols):
        """Mean bonds skipped and atoms dropped per built molecule, to size the rewriting."""
        built = [mol for mol in self._valid_mols(mols) if mol.HasProp("bonds_skipped")]

        if not built:
            return {"bonds_skipped": float("nan"), "atoms_dropped": float("nan")}

        return {name: float(np.mean([mol.GetIntProp(name) for mol in built]))
                for name in ("bonds_skipped", "atoms_dropped")}

    def mean_atoms(self, mols):
        """Mean heavy atom count of the molecules that exist."""
        counts = [mol.GetNumHeavyAtoms() for mol in self._valid_mols(mols)]

        return float(np.mean(counts)) if counts else float("nan")

    def lipinski_rule_of_five(self, mols):
        """Two readings of the same four conditions: the conventional gate, and the share."""
        rows = self._per_molecule(
            self._valid_mols(mols),
            lambda mol: [test(mol) for _, test in self.RULE_OF_FIVE])

        if not rows:
            return {"ro5": float("nan"), "ro5_conditions": float("nan")}

        met = np.array(rows)

        return {"ro5": float((met.sum(axis=1) >= 3).mean()),
                "ro5_conditions": float(met.mean())}

    def chemnet_activations(self, smiles, batch_size=128):
        """ChemNet penultimate activations for a list of SMILES, as [molecules, 512]."""
        from fcd.utils import SmilesDataset, todevice
        from torch.utils.data import DataLoader

        device = "cuda" if torch.cuda.is_available() else "cpu"
        loader = DataLoader(SmilesDataset(smiles), batch_size=batch_size)

        with todevice(self.ref_model, device), torch.no_grad():
            return np.vstack([
                self.ref_model(batch.transpose(1, 2).float().to(device)).cpu().numpy()
                for batch in loader
            ])

    @staticmethod
    def frechet_distance(activations_a, activations_b, eps=1e-6):
        """Frechet distance between two sets of activations, treated as Gaussians."""
        from scipy import linalg

        mu_a, mu_b = activations_a.mean(axis=0), activations_b.mean(axis=0)
        sigma_a = np.cov(activations_a, rowvar=False)
        sigma_b = np.cov(activations_b, rowvar=False)

        covmean = linalg.sqrtm(sigma_a.dot(sigma_b))

        # a singular product needs nudging off the boundary before the root is real
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_a.shape[0]) * eps
            covmean = linalg.sqrtm((sigma_a + offset).dot(sigma_b + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        difference = mu_a - mu_b

        return float(difference.dot(difference) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean))

    def frechet_chemnet_distance(self, mols, train_mols, reference=None):
        """FCD against train_mols, or against a reference from chemnet_activations."""
        generated_smiles = [
            Chem.MolToSmiles(mol, canonical=True)
            for mol in self._valid_mols(mols)
        ]

        if len(generated_smiles) < 2:
            return float("nan")

        if reference is None:
            train_smiles = [
                Chem.MolToSmiles(mol, canonical=True)
                for mol in self._valid_mols(train_mols)
            ]
            if not train_smiles:
                return float("nan")
            reference = self.chemnet_activations(train_smiles)

        return self.frechet_distance(self.chemnet_activations(generated_smiles), reference)

    def predicted_binding_affinity(self, predictions, success_threshold=7.0,
                                    total_count=None, keep=None):
        """Descriptive statistics of the on-target column, over EVERY scored molecule."""
        predictions = predictions.flatten()

        avg_pba = predictions.mean().item()

        cutoff = torch.quantile(predictions, 0.9)
        avg_top_10 = predictions[predictions >= cutoff].mean().item()

        qualifies = predictions >= success_threshold
        if keep is not None:
            qualifies = qualifies & torch.as_tensor(keep, dtype=torch.bool, device=qualifies.device)

        denominator = int(predictions.numel() if total_count is None else total_count)
        success_rate = int(qualifies.sum()) / denominator if denominator else float("nan")

        return avg_pba, avg_top_10, success_rate

    def delta_score(self, affinities, target_idx):
        on_target, off_target = self._target_partition(affinities, target_idx)

        return (on_target.mean() - off_target.mean()).item()

    def distributional_delta_score(self, affinities, target_idx, eps=SPREAD_FLOOR):
        """Signed KL between the on- and off-target distributions."""
        on_target, off_target = self._target_partition(affinities, target_idx)

        p_mu = on_target.mean()
        p_sigma = on_target.std(unbiased=False).clamp_min(eps)

        q_mu = off_target.mean()
        q_sigma = off_target.std(unbiased=False).clamp_min(eps)

        kl = torch.log(q_sigma / p_sigma) + (p_sigma**2 + (p_mu - q_mu)**2) / (2 * q_sigma**2) - 0.5

        return (torch.sign(p_mu - q_mu) * kl).item()

    def tanimoto_gap(self, mol_set_a, mol_set_b):
        mol_set_a = self._valid_mols(mol_set_a)
        mol_set_b = self._valid_mols(mol_set_b)

        fingerprints_a = [self.mfpgen.GetFingerprint(mol) for mol in mol_set_a]
        fingerprints_b = [self.mfpgen.GetFingerprint(mol) for mol in mol_set_b]

        similarity_matrix_a = np.array([BulkTanimotoSimilarity(fp, fingerprints_a) for fp in fingerprints_a])

        similarity_matrix_b = np.array([BulkTanimotoSimilarity(fp, fingerprints_b) for fp in fingerprints_b])

        similarity_matrix_cross = np.array([BulkTanimotoSimilarity(fp, fingerprints_b) for fp in fingerprints_a])

        if len(fingerprints_a) > 1:
            avg_within_a = similarity_matrix_a[~np.eye(len(fingerprints_a), dtype=bool)].mean()
        else:
            avg_within_a = np.nan

        if len(fingerprints_b) > 1:
            avg_within_b = similarity_matrix_b[~np.eye(len(fingerprints_b), dtype=bool)].mean()
        else:
            avg_within_b = np.nan

        avg_within = np.nanmean([avg_within_a, avg_within_b])
        avg_cross = similarity_matrix_cross.mean() if similarity_matrix_cross.size else np.nan

        return avg_within - avg_cross

    def tanimoto_gaps(self, libraries):
        """Attempt-weighted Tanimoto gaps over every valid generated molecule."""
        weighted = []
        for library in libraries:
            unique = {}
            for mol in self._valid_mols(library):
                fingerprint = self.mfpgen.GetFingerprint(mol)
                key = fingerprint.ToBitString()
                if key in unique:
                    unique[key][1] += 1
                else:
                    unique[key] = [fingerprint, 1]
            weighted.append(list(unique.values()))

        within = []
        for fingerprints in weighted:
            size = sum(frequency for _, frequency in fingerprints)
            denominator = size * (size - 1) / 2
            if not denominator:
                within.append(float("nan"))
                continue

            total = sum(frequency * (frequency - 1) / 2
                        for _, frequency in fingerprints)
            for i, (fingerprint, frequency) in enumerate(fingerprints):
                others = fingerprints[i + 1:]
                similarities = BulkTanimotoSimilarity(
                    fingerprint, [other for other, _ in others])
                total += sum(frequency * other_frequency * similarity
                                for similarity, (_, other_frequency)
                                in zip(similarities, others))
            within.append(float(total / denominator))

        count = len(weighted)
        cross = np.full((count, count), np.nan)
        for i in range(count):
            left_size = sum(frequency for _, frequency in weighted[i])
            for j in range(i + 1, count):
                right_size = sum(frequency for _, frequency in weighted[j])
                if not left_size or not right_size:
                    continue

                right_prints = [fingerprint for fingerprint, _ in weighted[j]]
                total = 0.0
                for fingerprint, frequency in weighted[i]:
                    similarities = BulkTanimotoSimilarity(fingerprint, right_prints)
                    total += sum(frequency * other_frequency * similarity
                                    for similarity, (_, other_frequency)
                                    in zip(similarities, weighted[j]))
                cross[i, j] = cross[j, i] = total / (left_size * right_size)

        gaps = []
        for i in range(count):
            partners = [j for j in range(count)
                        if j != i and np.isfinite(cross[i, j])]
            if not partners:
                gaps.append(float("nan"))
                continue

            gaps.append(float(np.nanmean(
                [np.nanmean([within[i], within[j]]) - cross[i, j] for j in partners])))

        return gaps

    def selectivity_effect(self, affinities, target_idx, spread_floor=LOSS_SPREAD_FLOOR):
        """Signed effect size of the on-target gap, on the objective's own scale."""
        on_target, off_target = self._target_partition(affinities, target_idx)
        row_gap = (on_target[:, None] - off_target).mean(dim=1)

        return (row_gap.mean() / row_gap.std(unbiased=False).clamp_min(spread_floor)).item()

    def selectivity_spread(self, affinities, target_idx):
        """Standard deviation of the per-molecule gap."""
        on_target, off_target = self._target_partition(affinities, target_idx)

        return (on_target[:, None] - off_target).mean(dim=1).std(unbiased=False).item()

    def directional_consistency(self, affinities, target_idx):
        """Per-molecule fraction of off-target affinities below its on-target affinity."""
        on_target, off_target = self._target_partition(affinities, target_idx)

        return (on_target[:, None] > off_target).float().mean().item()

    def hit_delta_score(self, affinities, target_idx, success_threshold=7.0, keep=None,
                        total_count=None, scored_keep=None):
        """Selectivity of the molecules that actually hit the target, and how many did."""
        on_target, off_target = self._target_partition(affinities, target_idx)

        potent = on_target >= success_threshold
        qualifies = potent
        if keep is not None:
            qualifies = qualifies & torch.as_tensor(keep, dtype=torch.bool, device=qualifies.device)

        hits = int(qualifies.sum())
        denominator = int(on_target.numel() if total_count is None else total_count)
        row_gap = on_target - off_target.mean(dim=1)

        scored_qualifies = potent
        if scored_keep is not None:
            scored_qualifies = scored_qualifies & torch.as_tensor(
                scored_keep, dtype=torch.bool, device=potent.device)
        effective_delta = (
            float(torch.where(scored_qualifies, row_gap, 0.0).sum()) / denominator
            if denominator else float("nan"))
        effective_count = int(scored_qualifies.sum())

        hit_delta = float(row_gap[qualifies].mean()) if hits else float("nan")

        # no hit_fraction here: it is hits / denominator, which is exactly success_rate
        return {"hit_delta_score": hit_delta,
                "effective_delta": effective_delta,
                "effective_count": effective_count,
                "hit_count": hits}

    def eval_affinities(self, affinities, target_idx, success_threshold=7.0, keep=None,
                        total_count=None, scored_keep=None):
        """Every metric that reads an oracle's predictions, and nothing else."""
        avg_pba, avg_top10, success_rate = self.predicted_binding_affinity(
            affinities[:, target_idx], success_threshold, total_count=total_count,
            keep=keep)

        return {
            **self.hit_delta_score(affinities, target_idx, success_threshold, keep,
                                    total_count=total_count,
                                    scored_keep=scored_keep),
            "average_pba": avg_pba,
            "average_top10": avg_top10,
            "success_rate": success_rate,
            "delta_score": self.delta_score(affinities, target_idx),
            "distributional_delta_score": self.distributional_delta_score(affinities, target_idx),
            "selectivity_effect": self.selectivity_effect(affinities, target_idx),
            "selectivity_spread": self.selectivity_spread(affinities, target_idx),
            "directional_consistency": self.directional_consistency(affinities, target_idx),
        }

    def eval_all(self, train_drugs, generated_drugs, comparison_sets=(), known_smiles=None):
        valid_pct, valid_mols = self.validity(generated_drugs)
        unique_pct, unique_mols = self.uniqueness(valid_mols)
        novel_pct = self.novelty(unique_mols, train_drugs, known=known_smiles)

        results = {
            "validity": valid_pct,
            "uniqueness": unique_pct,
            "novelty": novel_pct,
            "yield": valid_pct * unique_pct * novel_pct,
            "strict_validity": self.strict_validity(generated_drugs),
            **self.assembly_compromises(generated_drugs),
            "internal_diversity": self.internal_diversity(unique_mols),
            "mean_atoms": self.mean_atoms(generated_drugs),
            "qed": self.qed(unique_mols),
            "synthetic_accessibility": self.synthetic_accessibility(unique_mols),
            **self.lipinski_rule_of_five(unique_mols),
        }

        gaps = [self.tanimoto_gap(unique_mols, other) for other in comparison_sets if other]
        gaps = [gap for gap in gaps if np.isfinite(gap)]

        if gaps:
            results["tanimoto_gap"] = float(np.mean(gaps))

        return results
