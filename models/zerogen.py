# ZeroGEN (Chen et al., Bioinformatics 41(11), btaf572, 2025)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.chemical_tools import to_mols
from utils.tokenizer import SPECIAL_TOKENS

from .components import autoregressive_sample, causal_attention_mask, forbidden_generation_ids


def tie_except_self_attn(source, target, skip_key='self_attn'):
    for name, child in source._modules.items():
        if name == skip_key or name not in target._modules:
            continue
        if hasattr(child, 'weight'):
            target._modules[name] = child
        else:
            tie_except_self_attn(child, target._modules[name], skip_key)

class Attention(nn.Module):
    def __init__(self, d_model, nhead, dropout, is_cross=False, is_causal=False):
        super().__init__()
        if d_model % nhead:
            raise ValueError(f"d_model {d_model} is not divisible by nhead {nhead}")

        self.nhead = nhead
        self.is_cross = is_cross
        self.is_causal = is_causal

        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, memory=None, key_padding_mask=None, memory_padding_mask=None):
        source = memory if self.is_cross else x
        padding = memory_padding_mask if self.is_cross else key_padding_mask

        batch, length, width = x.shape
        source_length = source.size(1)
        head_width = width // self.nhead

        def heads(projection, tensor, n):
            return projection(tensor).view(batch, n, self.nhead, head_width).transpose(1, 2)

        q = heads(self.query, x, length)
        k = heads(self.key, source, source_length)
        v = heads(self.value, source, source_length)

        attention = (q @ k.transpose(-2, -1)) / math.sqrt(head_width)

        if self.is_causal and not self.is_cross:
            attention = attention.masked_fill(
                causal_attention_mask(length, x.device), float('-inf'))
        if padding is not None:
            attention = attention.masked_fill(padding[:, None, None, :], float('-inf'))

        attention = self.attn_drop(torch.softmax(attention, dim=-1))

        out = (attention @ v).transpose(1, 2).reshape(batch, length, width)
        return self.resid_drop(self.proj(out))


class Block(nn.Module):
    """Pre-norm block whose cross-attention runs only when a memory is supplied."""
    def __init__(self, d_model, nhead, dropout, is_causal=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.self_attn = Attention(d_model, nhead, dropout, is_causal=is_causal)
        self.cross_attn = Attention(d_model, nhead, dropout, is_cross=True)
        # release: 4 * ninp with GELU
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x, memory=None, key_padding_mask=None, memory_padding_mask=None):
        x = self.ln1(x)
        x = x + self.self_attn(x, key_padding_mask=key_padding_mask)

        if memory is not None:
            x = x + self.cross_attn(x, memory, memory_padding_mask=memory_padding_mask)

        return x + self.mlp(self.ln2(x))


class MolTransformer(nn.Module):
    """One molecular stack, made the decoder rather than the encoder by is_causal."""
    def __init__(self, vocab_size, d_model, nhead, depth, dropout, max_length,
                    pad_id, is_causal=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # release: learned positions, not a sinusoidal matrix
        self.pos_emb = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, nhead, dropout, is_causal=is_causal) for _ in range(depth))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, memory=None, key_padding_mask=None, memory_padding_mask=None):
        x = self.token_embedding(tokens)
        x = self.drop(x + self.pos_emb[:, :tokens.size(1), :])

        for block in self.blocks:
            x = block(x, memory, key_padding_mask=key_padding_mask, memory_padding_mask=memory_padding_mask)

        x = self.ln_f(x)
        # position 0 is the pooled representation, as in `mol_feature = x[:, 0]`
        return self.head(x), x[:, 0]


class ZeroGEN(nn.Module):
    def __init__(self, tokenizer, protein_dim, affinity_min=0.0, affinity_max=1.0,
                    d_model=128, nhead=4, depth=4, dropout=0.2, max_length=128,
                    queue_size=8192, momentum=0.995, temperature=0.07, alpha=0.2,
                    w_pbld=1.0, w_plcl=1.0, w_plip=1.0):
        super().__init__()

        vocab_size = tokenizer.vocab_size

        self.tokenizer = tokenizer
        # upstream min-max scales affinity into [0, 1] before training
        self.affinity_min = affinity_min
        self.affinity_span = max(affinity_max - affinity_min, 1e-6)
        self.max_length = max_length
        self.momentum = momentum
        self.queue_size = queue_size
        self.alpha = alpha
        self.epoch = 0
        self.w_pbld, self.w_plcl, self.w_plip = w_pbld, w_plcl, w_plip

        self.vocab_size = vocab_size
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.unk_id = tokenizer.unk_id
        self.mask_id = tokenizer.mask_id
        self.enc_id = tokenizer.enc_id

        self.first_char_id = len(SPECIAL_TOKENS)
        self.forbidden_ids = forbidden_generation_ids(tokenizer)

        self.protein_projection = nn.Linear(protein_dim, d_model)

        stack = dict(vocab_size=vocab_size, d_model=d_model, nhead=nhead, depth=depth,
                        dropout=dropout, max_length=max_length, pad_id=self.pad_id)
        self.mol_encoder = MolTransformer(**stack, is_causal=False)
        self.mol_decoder = MolTransformer(**stack, is_causal=True)
        tie_except_self_attn(self.mol_encoder, self.mol_decoder, skip_key='self_attn')

        # PLCL projections, with momentum copies feeding the queues
        self.mol2emb = nn.Linear(d_model, d_model)
        self.prot2emb = nn.Linear(d_model, d_model)
        self.mol_encoder_m = MolTransformer(**stack, is_causal=False)
        self.mol2emb_m = nn.Linear(d_model, d_model)
        self.prot2emb_m = nn.Linear(d_model, d_model)
        self.protein_projection_m = nn.Linear(protein_dim, d_model)

        self.model_pairs = [(self.mol_encoder, self.mol_encoder_m),
                            (self.protein_projection, self.protein_projection_m),
                            (self.mol2emb, self.mol2emb_m),
                            (self.prot2emb, self.prot2emb_m)]
        self._copy_to_momentum()

        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())
        self.temperature_range = (1e-2, 1.0)

        self.register_buffer('mol_queue', F.normalize(torch.randn(d_model, queue_size), dim=0))
        self.register_buffer('prot_queue', F.normalize(torch.randn(d_model, queue_size), dim=0))

        self.register_buffer('mol_idx_queue', torch.full((1, queue_size), -1, dtype=torch.long))
        self.register_buffer('prot_idx_queue', torch.full((1, queue_size), -1, dtype=torch.long))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.matching_head = nn.Sequential(nn.Linear(d_model, d_model), nn.Linear(d_model, 2))
        self.affinity_head = nn.Sequential(nn.Linear(d_model, d_model), nn.Linear(d_model, 1))

    @property
    def temperature(self):
        """The learned contrastive temperature, kept off zero."""
        low, high = self.temperature_range
        return self.log_temperature.exp().clamp(low, high)

    @torch.no_grad()
    def _copy_to_momentum(self):
        for online, target in self.model_pairs:
            target.load_state_dict(online.state_dict())
            for parameter in target.parameters():
                parameter.requires_grad = False

    @torch.no_grad()
    def sync_decoder_self_attention(self):
        """Start the causal stack from the bidirectional MLM solution, as upstream does."""
        for encoder_block, decoder_block in zip(self.mol_encoder.blocks, self.mol_decoder.blocks):
            decoder_block.self_attn.load_state_dict(encoder_block.self_attn.state_dict())

    @torch.no_grad()
    def _update_momentum(self):
        for online, target in self.model_pairs:
            for p_online, p_target in zip(online.parameters(), target.parameters()):
                p_target.mul_(self.momentum).add_(p_online, alpha=1 - self.momentum)

    @torch.no_grad()
    def _enqueue(self, mol_keys, prot_keys, mol_idx, prot_idx):
        batch = mol_keys.size(0)
        ptr = int(self.queue_ptr)

        indices = (torch.arange(batch, device=mol_keys.device) + ptr) % self.queue_size
        self.mol_queue[:, indices] = mol_keys.T
        self.prot_queue[:, indices] = prot_keys.T
        self.mol_idx_queue[:, indices] = mol_idx.view(1, -1)
        self.prot_idx_queue[:, indices] = prot_idx.view(1, -1)
        self.queue_ptr[0] = (ptr + batch) % self.queue_size

    def encode_protein(self, prot_emb):
        """Cached embedding -> [batch, 1, d_model], a length-one memory."""
        return self.protein_projection(prot_emb.float()).unsqueeze(1)

    def encode_molecule(self, tokens, memory=None, memory_padding_mask=None):
        return self.mol_encoder(tokens, memory,
                                key_padding_mask=tokens.eq(self.pad_id),
                                memory_padding_mask=memory_padding_mask)

    def pbld_loss(self, protein_memory, tokens):
        logits, _ = self.mol_decoder(tokens[:, :-1], protein_memory, key_padding_mask=tokens[:, :-1].eq(self.pad_id))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1), ignore_index=self.pad_id)

    def plcl_loss(self, mol_feature, prot_feature, tokens, prot_emb, drug_id, protein_id):
        mol_q = F.normalize(self.mol2emb(mol_feature), dim=-1)
        prot_q = F.normalize(self.prot2emb(prot_feature), dim=-1)

        drug_id = drug_id.view(-1, 1)
        protein_id = protein_id.view(-1, 1)

        mol_ids_all = torch.cat([drug_id.t(), self.mol_idx_queue.clone().detach()], dim=1)
        prot_ids_all = torch.cat([protein_id.t(), self.prot_idx_queue.clone().detach()], dim=1)

        hard_m2p = torch.eq(drug_id, mol_ids_all).float()
        hard_m2p /= hard_m2p.sum(1, keepdim=True)
        hard_p2m = torch.eq(protein_id, prot_ids_all).float()
        hard_p2m /= hard_p2m.sum(1, keepdim=True)

        with torch.no_grad():
            if self.training:
                self._update_momentum()

            _, mol_feature_m = self.mol_encoder_m(tokens, key_padding_mask=tokens.eq(self.pad_id))
            mol_k = F.normalize(self.mol2emb_m(mol_feature_m), dim=-1)
            prot_k = F.normalize(self.prot2emb_m(self.protein_projection_m(prot_emb.float())), dim=-1)

            mol_all = torch.cat([mol_k.t(), self.mol_queue.clone().detach()], dim=1)
            prot_all = torch.cat([prot_k.t(), self.prot_queue.clone().detach()], dim=1)

            alpha = self.alpha if self.epoch > 0 else 0.0
            soft_m2p = F.softmax(mol_k @ prot_all / self.temperature, dim=1)
            soft_p2m = F.softmax(prot_k @ mol_all / self.temperature, dim=1)

            target_m2p = alpha * soft_m2p + (1 - alpha) * hard_m2p
            target_p2m = alpha * soft_p2m + (1 - alpha) * hard_p2m

        sim_m2p = mol_q @ prot_all / self.temperature
        sim_p2m = prot_q @ mol_all / self.temperature

        loss = (-(F.log_softmax(sim_m2p, dim=1) * target_m2p).sum(1).mean()
                - (F.log_softmax(sim_p2m, dim=1) * target_p2m).sum(1).mean()) / 2

        if self.training:
            self._enqueue(mol_k, prot_k, drug_id, protein_id)

        return loss

    def plip_loss(self, tokens, mol_feature, prot_feature, protein_memory, affinity,
                    drug_id, protein_id):
        """Matching against hard negatives, plus affinity regression."""
        tokens = tokens.clone()
        tokens[:, 0] = self.enc_id

        _, positive_feature = self.encode_molecule(tokens, protein_memory)

        with torch.no_grad():
            drug_id = drug_id.view(-1, 1)
            protein_id = protein_id.view(-1, 1)

            sim_m2p = mol_feature @ prot_feature.t() / self.temperature
            sim_p2m = prot_feature @ mol_feature.t() / self.temperature

            weights_m2p = F.softmax(sim_m2p, dim=1).masked_fill(torch.eq(drug_id, drug_id.t()), 0)
            weights_p2m = F.softmax(sim_p2m, dim=1).masked_fill(torch.eq(protein_id, protein_id.t()), 0)

            weights_m2p = self._or_uniform(weights_m2p)
            weights_p2m = self._or_uniform(weights_p2m)

            protein_negative = torch.multinomial(weights_m2p, 1).squeeze(1)
            molecule_negative = torch.multinomial(weights_p2m, 1).squeeze(1)

        negative_tokens = torch.cat([tokens, tokens[molecule_negative]], dim=0)
        negative_memory = torch.cat([protein_memory[protein_negative], protein_memory], dim=0)
        _, negative_feature = self.encode_molecule(negative_tokens, negative_memory)

        logits = self.matching_head(torch.cat([positive_feature, negative_feature], dim=0))
        target = torch.cat([
                    torch.ones(tokens.size(0), dtype=torch.long, device=logits.device),
                    torch.zeros(2 * tokens.size(0), dtype=torch.long, device=logits.device)
                ])
        matching = F.cross_entropy(logits, target)

        predicted = self.affinity_head(positive_feature).squeeze(-1)
        scaled = (affinity - self.affinity_min) / self.affinity_span
        regression = F.mse_loss(predicted, scaled)

        return (matching / 3 + regression) / 2, predicted

    @staticmethod
    def _or_uniform(weights):
        empty = weights.sum(1, keepdim=True) <= 0
        return torch.where(empty, torch.ones_like(weights), weights)

    def compute_loss(self, data):
        tokens = data.drug_tokens

        protein_memory = self.encode_protein(data.prot_emb)
        prot_feature = protein_memory.squeeze(1)
        _, mol_feature = self.encode_molecule(tokens)

        pbld = self.pbld_loss(protein_memory, tokens)
        plcl = self.plcl_loss(mol_feature, prot_feature, tokens, data.prot_emb, data.drug_id, data.protein_id)
        plip, _ = self.plip_loss(tokens,
                                F.normalize(self.mol2emb(mol_feature), dim=-1),
                                F.normalize(self.prot2emb(prot_feature), dim=-1),
                                protein_memory, data.affinity.float(),
                                data.drug_id, data.protein_id)

        total = self.w_pbld * pbld + self.w_plcl * plcl + self.w_plip * plip
        return total, {'loss': total.item(), 'pbld': pbld.item(), 'plcl': plcl.item(), 'plip': plip.item()}

    def pretrain_loss(self, data, mask_probability=0.3):
        """Masked-token pretraining loss, minimised before joint training."""
        tokens = data.drug_tokens

        # never corrupt bos, eos or padding: only the molecule itself
        maskable = ~(tokens.eq(self.pad_id) | tokens.eq(self.bos_id) | tokens.eq(self.eos_id))
        selected = (torch.rand_like(tokens, dtype=torch.float) < mask_probability) & maskable
        split = torch.rand_like(tokens, dtype=torch.float)

        corrupted = tokens.clone()
        corrupted[selected & (split < 0.8)] = self.mask_id

        # a real molecular token, never a special one
        randomised = selected & (split >= 0.8) & (split < 0.9)
        corrupted[randomised] = torch.randint(self.first_char_id, self.vocab_size, (int(randomised.sum()),), device=tokens.device)

        logits, _ = self.encode_molecule(corrupted)

        # the uncorrupted 10% is scored too, so the encoder cannot assume a token is right
        labels = tokens.masked_fill(~selected, -100)

        # cross_entropy over an all-ignored batch returns NaN, but selecting nothing is a no-op
        if not bool(selected.any()):
            loss = logits.sum() * 0.0
            return loss, {'loss': 0.0, 'masked_tokens': 0}

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

        return loss, {'loss': loss.item(), 'masked_tokens': int(selected.sum())}

    def on_epoch_end(self):
        """Momentum distillation switches on after the first epoch."""
        self.epoch += 1

    @torch.no_grad()
    def generate(self, prot_emb, prot_seq=None, seed=0, temperature=1.0, num_atoms=None):
        """Zero-shot: decode ligands straight from a protein embedding."""
        was_training = self.training
        self.eval()

        try:
            protein_memory = self.encode_protein(prot_emb)

            def next_logits(tokens, memory):
                logits, _ = self.mol_decoder(tokens, memory, key_padding_mask=tokens.eq(self.pad_id))
                return logits[:, -1]

            tokens = autoregressive_sample(
                next_logits,
                n_samples=protein_memory.size(0),
                bos_id=self.bos_id, eos_id=self.eos_id, pad_id=self.pad_id,
                max_length=self.max_length,
                device=prot_emb.device, conditions=(protein_memory,),
                temperature=temperature, seed=seed, forbidden_ids=self.forbidden_ids,
                require_ids=self.tokenizer.content_ids)
        finally:
            self.train(was_training)

        return {'seq': to_mols(self.tokenizer.decode_batch(tokens))}
