import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_batch

from .chemical_tools import DECODED_ATOM_BLOCKS, dense_bond_types, split_atom_features


class MarginRankLoss(nn.Module):
    """Hinge on a score gap, weighted by the affinity gap that ordered the pair."""
    def __init__(self, margin, eps=1e-6):
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, positive_score, negative_score, affinity_gap):
        hinge = F.relu(self.margin - (positive_score - negative_score))
        weight = affinity_gap.detach()

        return (weight * hinge).sum() / weight.sum().clamp_min(self.eps)

class DistributionalDeltaLoss(nn.Module):
    """Every drug above its own panel, and the batch separated from the panel as a whole."""
    def __init__(self, direction_weight, effect_margin, mean_margin, eps=1e-6, spread_floor=1e-3):
        super().__init__()

        if not 0 <= direction_weight <= 1:
            raise ValueError("direction_weight must be between 0 and 1")

        self.direction_weight = direction_weight
        self.effect_margin = effect_margin
        self.mean_margin = mean_margin
        self.eps = eps
        self.spread_floor = spread_floor

    @staticmethod
    def _empty():
        return {name: float("nan") for name in ("directional", "separation", "gap_mean", "gap_std", "effect")}

    def forward(self, scores, complete, target_idx=0):
        on_target = scores[:, target_idx]
        off_target = torch.cat([scores[:, :target_idx], scores[:, target_idx + 1:]], dim=1)

        gaps = on_target[:, None] - off_target

        weight = complete.float()
        rows = weight.sum()

        if not bool(rows > 0):
            return scores.sum() * 0.0, {**self._empty(), "complete": 0.0}

        total = rows.clamp_min(self.eps)

        row_gap = gaps.mean(dim=1)
        directional_loss = (F.relu(self.mean_margin - row_gap) * weight).sum() / total

        mean = (row_gap * weight).sum() / total
        if bool(rows > 1):
            spread = ((row_gap - mean).pow(2) * weight).sum() / total
            deviation = (spread + self.eps).sqrt().clamp_min(self.spread_floor)
            effect = mean / deviation
            separation_loss = F.relu(self.effect_margin - effect)
        else:
            deviation = effect = separation_loss = scores.sum() * 0.0

        loss = (1 - self.direction_weight) * separation_loss + self.direction_weight * directional_loss

        measurable = bool(rows > 1)
        directional, separation, gap_mean, gap_std, effect_size, share = torch.stack(
            [directional_loss, separation_loss, mean, deviation, effect, rows / weight.numel()]).detach().tolist()

        return loss, {"directional": directional,
                        "separation": separation if measurable else float("nan"),
                        "gap_mean": gap_mean,
                        "gap_std": gap_std if measurable else float("nan"),
                        "effect": effect_size if measurable else float("nan"),
                        "complete": share}

class SequenceReconstructionLoss(nn.Module):
    """Teacher-forced token NLL, shifting so logits[:, :-1] predicts tokens[:, 1:]."""
    def __init__(self, pad_id=0):
        super().__init__()
        self.pad_id = pad_id

    def forward(self, logits, tokens):
        logits, target = logits[:, :-1], tokens[:, 1:]
        token_nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
            ignore_index=self.pad_id, reduction="none").view(target.shape)

        real = target.ne(self.pad_id)

        return (token_nll * real).sum() / real.sum().clamp_min(1)

class GNNReconstructionLoss(nn.Module):
    """Masked NLL of a decoded graph: atom existence, the decoded atom blocks, and bonds."""
    def __init__(self, no_bond_weight=1.0):
        super().__init__()
        self.no_bond_weight = no_bond_weight

    def _edge_class_weight(self, edge_logits):
        # Down-weight the no-bond class
        if self.no_bond_weight == 1.0:
            return None

        weight = torch.ones(edge_logits.size(-1), device=edge_logits.device, dtype=edge_logits.dtype)
        weight[0] = self.no_bond_weight

        return weight

    def terms(self, output, data_batch):
        """Every component as a per-molecule NLL [batch], before weighting."""
        num_nodes = output["node_logits"].size(1)

        nodes, real = to_dense_batch(data_batch.x, data_batch.batch, max_num_nodes=num_nodes)
        target = split_atom_features(nodes)
        bonds = dense_bond_types(data_batch, num_nodes)

        atom_mask = real.float()
        # a bond needs both ends, and an atom does not bond to itself
        pair_mask = (real.unsqueeze(1) & real.unsqueeze(2)).float()
        pair_mask = pair_mask * (1 - torch.eye(num_nodes, device=real.device))

        terms = {
            "node": F.binary_cross_entropy_with_logits(output["node_logits"], atom_mask, reduction="none").sum(-1),

            # the matrix holds each undirected bond twice
            "edge": 0.5 * (F.cross_entropy(
                output["edge_logits"].permute(0, 3, 1, 2), bonds,
                weight=self._edge_class_weight(output["edge_logits"]),
                reduction="none") * pair_mask).sum((1, 2)),
        }

        for name, _ in DECODED_ATOM_BLOCKS:
            block_nll = F.cross_entropy(output["atom_logits"][name].transpose(1, 2), target[name].argmax(-1), reduction="none")
            terms[name] = (block_nll * atom_mask).sum(-1)

        return terms

    def forward(self, output, data_batch):
        terms = self.terms(output, data_batch)

        # per atom, so the term does not scale with molecule size and drown the rest
        atoms = torch.bincount(data_batch.batch).clamp_min(1)

        edge = terms.pop("edge") / (atoms * (atoms - 1) / 2).clamp_min(1)

        return (sum(terms.values()) / atoms + edge).mean()

class CompositeLoss(nn.Module):
    """The full SelVAEGen objective: reconstruction, KL, and the three selectivity terms."""
    def __init__(self, pad_id=0, affinity_rank_margin=1.0, selectivity_rank_margin=1.0,
                    distribution_mean_margin=1.0, distribution_effect_margin=1.0,
                    direction_weight=0.5, no_bond_weight=1.0,
                    w_seq=1.0, w_graph=1.0, w_kl_mean=0.05, w_kl_var=1.0,
                    w_affinity=1.0, w_selectivity=1.0, w_distributional=1.0,
                    kl_eps=1e-6):
        super().__init__()
        self.seq_recon = SequenceReconstructionLoss(pad_id=pad_id)
        self.gnn_recon = GNNReconstructionLoss(no_bond_weight=no_bond_weight)
        self.affinity_rank = MarginRankLoss(affinity_rank_margin)
        self.selectivity_rank = MarginRankLoss(selectivity_rank_margin)
        self.distributional = DistributionalDeltaLoss(direction_weight, distribution_effect_margin, distribution_mean_margin)

        self.kl_eps = kl_eps
        self.weights = {"kl_mean": w_kl_mean, "kl_var": w_kl_var,
                        "recon_seq": w_seq, "recon_graph": w_graph,
                        "affinity": w_affinity, "selectivity": w_selectivity,
                        "distributional": w_distributional}

    def _weighted_mean(self, per_row, weights):
        if weights is None:
            return per_row.mean()

        return (weights * per_row).sum() / weights.sum().clamp_min(self.kl_eps)

    def kl_divergence(self, mu, logvar, prior_mu, weights=None):
        """The two halves of KL(q || N(m(P), I)), which pull in opposite directions."""
        mean_part = (0.5 * (mu - prior_mu).pow(2)).sum(-1)
        var_part = (0.5 * (logvar.exp() - 1 - logvar)).sum(-1)

        return (self._weighted_mean(mean_part, weights), self._weighted_mean(var_part, weights))

    def forward(self, outputs, mu, logvar, data_batch, affinity_ranking,
                selectivity_ranking, scores, prior_mu, complete, target_idx=0,
                kl_weights=None):
        kl_mean, kl_var = self.kl_divergence(mu, logvar, prior_mu, weights=kl_weights)

        parts = {"kl_mean": kl_mean, "kl_var": kl_var,
                    "affinity": self.affinity_rank(*affinity_ranking),
                    "selectivity": self.selectivity_rank(*selectivity_ranking)}

        parts["distributional"], pieces = self.distributional(scores, complete, target_idx)
        diagnostics = {f"dist_{name}": value for name, value in pieces.items()}

        if "seq" in outputs:
            parts["recon_seq"] = self.seq_recon(outputs["seq"], data_batch.drug_tokens)
        if "graph" in outputs:
            parts["recon_graph"] = self.gnn_recon(outputs["graph"], data_batch)

        total = torch.zeros((), device=mu.device)
        for name, weight in self.weights.items():
            if weight != 0 and name in parts:
                total = total + weight * parts[name]

        # one device sync for every scalar at once, rather than one per term per step
        present = [name for name in self.weights if name in parts]
        scalars = torch.stack([total] + [parts[name] for name in present]).detach().tolist()
        values = dict(zip(present, scalars[1:]))

        metrics = {}
        for name, weight in self.weights.items():
            metrics[name] = values.get(name, float("nan"))
            metrics[f"weighted_{name}"] = weight * values[name] if name in values else 0.0

        return total, {"loss": scalars[0], **metrics, **diagnostics}
