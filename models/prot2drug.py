# Prot2Drug (Creanza et al., JCIM 65, 1258-1277, 2025)
import math

import torch
import torch.nn as nn

from utils.chemical_tools import to_mols

from .components import SinusoidalPositionalEncoding, autoregressive_sample, causal_attention_mask, forbidden_generation_ids


class Prot2Drug(nn.Module):
    """Decoder-only conditional SMILES model, the protein vector is a length-one memory."""
    def __init__(self, tokenizer, protein_dim, nhead=32, decoder_layers=2,
                    dim_feedforward=2048, dropout=0.1, max_length=128):
        super().__init__()

        # the paper sets d_m to the protein embedding width, so the vector is used raw
        d_model = protein_dim

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.d_model = d_model
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.forbidden_ids = forbidden_generation_ids(tokenizer)

        self.smiles_embedding = nn.Embedding(tokenizer.vocab_size, d_model, padding_idx=self.pad_id)
        self.position = SinusoidalPositionalEncoding(d_model, max_length, dropout=0.0)

        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                            dim_feedforward=dim_feedforward, dropout=dropout,
                                            activation="relu", batch_first=True,
                                            norm_first=False)
        self.decoder = nn.TransformerDecoder(layer, decoder_layers)
        self.out = nn.Linear(d_model, tokenizer.vocab_size, bias=False)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def decode(self, prot_emb, drug_tokens):
        memory = prot_emb.float().unsqueeze(1)
        # Vaswani scales the embedding before the positional signal is added
        hidden = self.position(self.smiles_embedding(drug_tokens) * math.sqrt(self.d_model))
        hidden = self.decoder(
            hidden, memory,
            tgt_mask=causal_attention_mask(drug_tokens.size(1), drug_tokens.device),
            tgt_key_padding_mask=drug_tokens.eq(self.pad_id))

        return self.out(hidden)

    def compute_loss(self, batch):
        tokens = batch.drug_tokens
        logits = self.decode(batch.prot_emb, tokens[:, :-1])
        loss = self.loss(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))

        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def generate(self, prot_emb, prot_seq=None, seed=0, temperature=1.0, num_atoms=None):
        """One molecule per row of prot_emb."""
        training = self.training
        self.eval()                           # dropout off, or the seed does not hold
        try:
            tokens = autoregressive_sample(
                lambda drug_tokens, prot: self.decode(prot, drug_tokens)[:, -1],
                n_samples=prot_emb.size(0),
                bos_id=self.bos_id, eos_id=self.eos_id, pad_id=self.pad_id,
                max_length=self.max_length, device=prot_emb.device,
                conditions=(prot_emb.float(),), temperature=temperature, seed=seed,
                forbidden_ids=self.forbidden_ids, require_ids=self.tokenizer.content_ids)
        finally:
            self.train(training)

        return {"seq": to_mols(self.tokenizer.decode_batch(tokens))}
