import numpy as np
import pandas as pd
import torch

from .losses import CompositeLoss


def orient(score_a, score_b, gap):
    """Order a pair so the first element is the stronger binder."""
    stronger = gap >= 0

    return (torch.where(stronger, score_a, score_b),
            torch.where(stronger, score_b, score_a),
            gap.abs())


class SelectivityPool:
    """The other targets each drug was measured against, within one split."""
    def __init__(self, table, seed=0):
        self.rng = np.random.default_rng(seed)
        self.by_drug = {int(drug): group[["protein_id", "affinity"]].to_numpy()
                        for drug, group in table.groupby("drug_id")}

    def partners(self, drug_ids, protein_ids, rng=None):
        """One different measured target per row, and which rows have one at all."""
        rng = self.rng if rng is None else rng
        proteins = np.zeros(len(drug_ids), dtype=np.int64)
        affinities = np.zeros(len(drug_ids), dtype=np.float32)
        found = np.zeros(len(drug_ids), dtype=bool)

        for row, (drug, protein) in enumerate(zip(drug_ids, protein_ids)):
            measured = self.by_drug.get(int(drug))
            if measured is None:
                continue

            other = measured[measured[:, 0] != protein]

            if len(other):
                proteins[row], affinities[row] = other[rng.integers(len(other))]
                found[row] = True

        return proteins, affinities, found


class TargetPanel:
    """Per row, a panel of train targets the drug is measured to bind worse than its own."""
    def __init__(self, table, protein_embs, num_drugs, num_proteins, panel_size=3, device="cpu"):
        self.panel_size = panel_size
        self.device = device
        self.embeddings = protein_embs.to(device)

        drugs = np.sort(table["drug_id"].unique())
        proteins = np.sort(table["protein_id"].unique())

        self.protein_ids = torch.as_tensor(proteins)
        # sized by the full vocabulary, so an unseen id lands on -1 rather than needing a clamp
        self.drug_pos = self._lookup(drugs, num_drugs)
        self.protein_pos = self._lookup(proteins, num_proteins)

        self.affinity = torch.full((len(drugs), len(proteins)), float("nan"))
        self.affinity[
                    self.drug_pos[table["drug_id"].to_numpy().copy()],
                    self.protein_pos[table["protein_id"].to_numpy().copy()]
                    ] = torch.as_tensor(table["affinity"].to_numpy().copy(), dtype=torch.float32)

        measured = self.affinity[self.affinity.isfinite()]
        self.affinity_min, self.affinity_max = float(measured.min()), float(measured.max())

        self.keys = None

    def freeze(self, epoch, seed=0):
        """Fix every drug's panel for the whole epoch, on a stream of its own."""
        generator = torch.Generator().manual_seed(seed + 1000003 * int(epoch))
        self.keys = torch.rand(self.affinity.shape, generator=generator)

    @staticmethod
    def _lookup(ids, size):
        """Row of each id in the split's own ordering, -1 where it never appears."""
        table = torch.full((size,), -1, dtype=torch.long)
        table[torch.as_tensor(ids)] = torch.arange(len(ids))

        return table

    def sample(self, drug_ids, protein_ids, generator=None):
        """Panel ids [B, panel_size], the row's own affinity, and two row masks."""
        drug_ids, protein_ids = drug_ids.cpu(), protein_ids.cpu()

        rows = self.drug_pos[drug_ids]
        known = rows >= 0

        measured = torch.full((len(rows), len(self.protein_ids)), float("nan"))
        measured[known] = self.affinity[rows[known]]

        columns = self.protein_pos[protein_ids]
        seen = known & (columns >= 0)
        own = torch.full((len(rows),), float("nan"))
        own[seen] = measured[seen, columns[seen]]

        eligible = measured < own[:, None]
        eligible &= ~self.protein_ids[None, :].eq(protein_ids[:, None])

        if self.keys is None:
            keys = torch.rand(len(rows), len(self.protein_ids), generator=generator)
        else:
            # a drug outside the pool has no eligible target either, so its keys are moot
            keys = torch.zeros(len(rows), len(self.protein_ids))
            keys[known] = self.keys[rows[known]]

        panel = self.protein_ids[keys.masked_fill(~eligible, 2.0).argsort(dim=1)[:, :self.panel_size]]

        complete = eligible.sum(1) >= self.panel_size

        return (panel.to(self.device), own.to(self.device), seen.to(self.device), complete.to(self.device))


class SelectivityObjective:
    """Every loss term for one batch, on one encode and one decode."""
    def __init__(self, panel, pool, criterion, seed=0, freeze_panel=False, eval_panel=None, eval_pool=None):
        self.panel = panel
        self.eval_panel = panel if eval_panel is None else eval_panel
        self.pool = pool
        self.eval_pool = pool if eval_pool is None else eval_pool
        self.criterion = criterion
        self.seed = seed
        self.freeze_panel = freeze_panel

    def set_epoch(self, epoch):
        """Redraw the frozen panels, if the objective was built to freeze them at all."""
        if self.freeze_panel:
            self.panel.freeze(epoch, seed=self.seed)

    @torch.no_grad()
    def geometry(self, model, sample=256):
        """Median spacing of the learned target centres against the prior's sqrt(2d)."""
        ids = self.panel.protein_ids
        if len(ids) > sample:
            picked = torch.randperm(len(ids), generator=torch.Generator().manual_seed(self.seed))
            ids = ids[picked[:sample]]

        centres = model.prior_mean(self.panel.embeddings[ids.to(self.panel.embeddings.device)])
        distances = torch.cdist(centres, centres)
        spacing = float(distances[~torch.eye(len(ids), dtype=torch.bool, device=distances.device)].median())

        return {"centre_spacing": spacing, "centre_spacing_ratio": spacing / (2 * centres.shape[1]) ** 0.5}

    @staticmethod
    def latent_score(model, mu, prot_emb):
        """Target compatibility: -||mu - m(P)||^2 / 2, on the same scale as the KL."""
        return -0.5 * (mu - model.prior_mean(prot_emb)).pow(2).sum(-1)

    def affinity_pairs(self, model, mu, prot_emb, protein_ids, affinities, seed=None):
        """Two drugs on one target, paired inside the batch so nothing is re-encoded."""
        candidates = (protein_ids[:, None].eq(protein_ids[None, :]) & ~affinities[:, None].eq(affinities[None, :])).float()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=candidates.device).manual_seed(seed)

        partner = torch.multinomial(candidates + 1e-12, 1, generator=generator).squeeze(1)
        gap = (affinities - affinities[partner]) * candidates.sum(1).gt(0)

        return orient(self.latent_score(model, mu, prot_emb), self.latent_score(model, mu[partner], prot_emb), gap)

    def selectivity_pairs(self, model, mu, prot_emb, drug_ids, protein_ids, affinities, rng=None, pool=None):
        """One drug on two targets; the partner is a lookup, so nothing is re-encoded."""
        proteins, partner_affinity, found = (self.pool if pool is None else pool).partners(
            drug_ids.cpu().numpy(), protein_ids.cpu().numpy(), rng)

        device = mu.device
        partner_emb = self.panel.embeddings[torch.as_tensor(proteins, device=device)]
        gap = (affinities - torch.as_tensor(partner_affinity, device=device)) \
                * torch.as_tensor(found, device=device)

        return orient(self.latent_score(model, mu, prot_emb), self.latent_score(model, mu, partner_emb), gap)

    def __call__(self, model, batch):
        mu, logvar = model.encode(batch)
        z = model.reparameterize(mu, logvar) if model.training else mu
        outputs = model.decode(z, batch)

        drug_ids = batch.drug_id.view(-1)
        protein_ids = batch.protein_id.view(-1)
        affinities = batch.affinity.view(-1).float()
        prot_emb = batch.prot_emb

        training = model.training
        generator = None if training else torch.Generator().manual_seed(self.seed)
        rng = None if training else np.random.default_rng(self.seed)
        source = self.panel if training else self.eval_panel
        pool = self.pool if training else self.eval_pool
        panel, own, seen, complete = source.sample(drug_ids, protein_ids, generator)

        scores = torch.stack(
            [self.latent_score(model, mu, prot_emb)]
            + [self.latent_score(model, mu, source.embeddings[column])
                for column in panel.t()], dim=1)

        span = max(self.panel.affinity_max - self.panel.affinity_min, 1e-6)
        kl_weights = torch.where(
            seen, (own.nan_to_num(0.0) - self.panel.affinity_min) / span,
            torch.full_like(own, 0.5)).clamp_min(0.0)

        return self.criterion(
            outputs, mu, logvar, batch,
            self.affinity_pairs(model, mu, prot_emb, protein_ids, affinities,
                                None if training else self.seed),
            self.selectivity_pairs(model, mu, prot_emb, drug_ids, protein_ids, affinities,
                                    rng, pool),
            scores, model.prior_mean(prot_emb), complete, kl_weights=kl_weights)


def build_objective(train_table, protein_embs, num_drugs, num_proteins, pad_id,
                    panel_size=3, seed=0, device="cpu", freeze_panel=False,
                    val_table=None, **loss_kwargs):
    """The criterion SelVAEGen trains against, with its split-specific lookups."""
    panel = TargetPanel(train_table, protein_embs, num_drugs, num_proteins, panel_size=panel_size, device=device)
    pool = SelectivityPool(train_table, seed=seed)

    eval_panel = eval_pool = None
    if val_table is not None and len(val_table):
        combined = pd.concat([train_table, val_table])
        eval_panel = TargetPanel(combined, protein_embs, num_drugs, num_proteins, panel_size=panel_size, device=device)
        eval_pool = SelectivityPool(combined, seed=seed)

    return SelectivityObjective(panel, pool, CompositeLoss(pad_id=pad_id, **loss_kwargs),
                                seed=seed, freeze_panel=freeze_panel,
                                eval_panel=eval_panel, eval_pool=eval_pool)
