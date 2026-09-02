import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch

from utils.chemical_tools import (assemble_molecule, ATOM_NUM_FEATURES, ATOM_SYMBOLS,
                                    BOND_NUM_FEATURES, BOND_TYPES,
                                    DECODED_ATOM_BLOCKS, to_mols)

from .components import FiLM, SinusoidalPositionalEncoding, autoregressive_sample, causal_attention_mask, forbidden_generation_ids


class GNNEncoder(nn.Module):
    def __init__(self, atom_dim: int = ATOM_NUM_FEATURES, edge_dim: int = BOND_NUM_FEATURES,
                    hidden_channels: int = 128, out_channels: int = 512,
                    num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(atom_dim, hidden_channels)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(GATv2Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=num_heads,
                concat=False,
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=False))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.atom_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.input_proj(x)

        for conv, norm in zip(self.layers, self.norms):
            h = conv(x, edge_index, edge_attr=edge_attr)
            h = F.leaky_relu(h)
            h = self.dropout(h)

            x = norm(x + h)

        return self.atom_norm(self.output_proj(x))

class GNNDecoder(nn.Module):
    """Decodes a latent into a whole molecular graph at once: atoms and every bond."""
    def __init__(self, latent_dim: int = 512, hidden_channels: int = 256,
                    num_layers: int = 4, num_heads: int = 8, ff_dim: int = 1024,
                    dropout: float = 0.1, max_nodes: int = 64,
                    num_edge_classes: int = len(BOND_TYPES), edge_rank: int = 32,
                    protein_dim: int = 128):
        super().__init__()
        self.max_nodes = max_nodes
        self.num_edge_classes = num_edge_classes  # includes class 0, "no bond"
        self.edge_rank = edge_rank

        self.latent_proj = nn.Linear(latent_dim, hidden_channels)
        self.node_query = nn.Parameter(torch.randn(1, max_nodes, hidden_channels) * 0.02)

        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_channels, nhead=num_heads, dim_feedforward=ff_dim,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(num_layers))
        self.norm = nn.LayerNorm(hidden_channels)

        self.protein_proj = nn.Linear(protein_dim, hidden_channels)
        self.films = nn.ModuleList(FiLM(hidden_channels, hidden_channels) for _ in range(num_layers))

        self.node_head = nn.Linear(hidden_channels, 1)
        self.atom_head = nn.Linear(hidden_channels, sum(size for _, size in DECODED_ATOM_BLOCKS))

        self.edge_left = nn.Linear(hidden_channels, self.num_edge_classes * edge_rank)
        self.edge_right = nn.Linear(hidden_channels, self.num_edge_classes * edge_rank)
        self.edge_bias = nn.Parameter(torch.zeros(self.num_edge_classes))

    def _resolve_num_nodes(self, data_batch, num_nodes):
        if num_nodes is None:
            # follow to_dense_batch, so the slots line up with the encoder-side targets
            num_nodes = (self.max_nodes if data_batch is None else int(torch.bincount(data_batch.batch).max()))

        if num_nodes > self.max_nodes:
            raise ValueError(f"the batch needs {num_nodes} node slots, but the decoder was built for {self.max_nodes}")

        return num_nodes

    def _edge_logits(self, h):
        batch, num_nodes, _ = h.shape
        shape = (batch, num_nodes, self.num_edge_classes, self.edge_rank)

        left = self.edge_left(h).view(shape)
        right = self.edge_right(h).view(shape)

        logits = torch.einsum("bicr,bjcr->bijc", left, right) + self.edge_bias

        # a bond has no direction
        return 0.5 * (logits + logits.transpose(1, 2))

    def forward(self, z, prot_emb, data_batch=None, num_nodes=None):
        num_nodes = self._resolve_num_nodes(data_batch, num_nodes)
        condition = self.protein_proj(prot_emb.float())

        h = self.node_query[:, :num_nodes] + self.latent_proj(z).unsqueeze(1)

        for depth, block in enumerate(self.blocks):
            h = block(h)
            h = self.films[depth](h, condition)

        h = self.norm(h)

        atom_logits = self.atom_head(h)
        blocks, offset = {}, 0
        for name, size in DECODED_ATOM_BLOCKS:
            blocks[name] = atom_logits[..., offset:offset + size]
            offset += size

        return {
            "node_logits": self.node_head(h).squeeze(-1),  # [B, N]
            "atom_logits": blocks,                         # {block: [B, N, size]}
            "edge_logits": self._edge_logits(h),           # [B, N, N, num_edge_classes]
        }

    @staticmethod
    def _pick(logits, temperature, generator):
        if temperature is None or temperature <= 0:
            return logits.argmax(-1)

        probabilities = torch.softmax(logits.reshape(-1, logits.size(-1)) / temperature, dim=-1)
        drawn = torch.multinomial(probabilities, num_samples=1, generator=generator)

        return drawn.view(logits.shape[:-1])

    @torch.no_grad()
    def generate(self, z, prot_emb, num_nodes=None, node_threshold=0.5, temperature=None, seed=None, target_atoms=None):
        """Decode latents into valence-bounded RDKit molecules."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device=z.device).manual_seed(seed)

        output = self(z, prot_emb, num_nodes=num_nodes)

        if target_atoms is None:
            keep = torch.sigmoid(output["node_logits"]) >= node_threshold
            empty = ~keep.any(dim=1)
            if empty.any():                       # never hand back an atomless molecule
                keep[empty, output["node_logits"][empty].argmax(dim=1)] = True
        else:
            wanted = torch.as_tensor(target_atoms, device=output["node_logits"].device)
            wanted = wanted.expand(output["node_logits"].shape[0]).clamp(1, output["node_logits"].shape[1])
            order = output["node_logits"].argsort(dim=1, descending=True)
            ranks = order.argsort(dim=1)
            keep = ranks < wanted.unsqueeze(1)
        symbols = self._pick(output["atom_logits"]["symbol"][..., :len(ATOM_SYMBOLS)], temperature, generator)
        chiral = self._pick(output["atom_logits"]["chiral"], temperature, generator)
        charge = self._pick(output["atom_logits"]["charge"], temperature, generator)

        # assemble_molecule ranks pairs by confidence
        edge_scores = torch.softmax(output["edge_logits"] / (temperature if temperature and temperature > 0 else 1.0), dim=-1)

        mols = []
        for row in range(z.size(0)):
            picked = keep[row].nonzero(as_tuple=True)[0]

            mols.append(assemble_molecule(
                [ATOM_SYMBOLS[int(symbols[row, position])] for position in picked],
                edge_scores[row][picked][:, picked].float().cpu().numpy(),
                chiral=[int(c) for c in chiral[row, picked]],
                charges=[int(c) for c in charge[row, picked]]))

        return mols


class SequenceEncoder(nn.Module):
    """Bidirectional transformer over drug tokens."""
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 4,
                    num_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.1,
                    max_length: int = 128, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = SinusoidalPositionalEncoding(d_model, max_length, dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

    def forward(self, tokens, padding_mask=None):
        if padding_mask is None:
            padding_mask = tokens.eq(self.pad_id)

        x = self.position(self.token_embedding(tokens))

        return self.encoder(x, src_key_padding_mask=padding_mask)  # [B, S, D]


class SequenceDecoder(nn.Module):
    """Decodes drug tokens by cross-attending to the latent and the target condition."""
    def __init__(self, vocab_size: int, tokenizer=None, latent_dim: int = 512,
                    d_model: int = 512, num_layers: int = 4, num_heads: int = 8,
                    ff_dim: int = 1024, dropout: float = 0.1, max_length: int = 128,
                    pad_id: int = 0, bos_id: int = 1, eos_id: int = 2,
                    protein_dim: int = 128):
        super().__init__()

        if tokenizer is not None:
            if vocab_size != tokenizer.vocab_size:
                raise ValueError(f"vocab_size={vocab_size}, but tokenizer has {tokenizer.vocab_size} tokens")
            pad_id, bos_id, eos_id = tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.forbidden_generation_ids = (forbidden_generation_ids(tokenizer) if tokenizer is not None else (pad_id, bos_id))

        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = SinusoidalPositionalEncoding(d_model, max_length, dropout)

        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

        self.protein_proj = nn.Linear(protein_dim, d_model)
        self.films = nn.ModuleList(FiLM(d_model, d_model) for _ in range(num_layers))

    def _memory(self, z, prot_emb):
        """The cross-attention memory, and the condition driving the FiLM layers."""
        memory = self.latent_proj(z).unsqueeze(1)
        condition = self.protein_proj(prot_emb.float())

        return torch.cat([memory, condition.unsqueeze(1)], dim=1), condition

    def _decode(self, memory, tokens, condition, padding_mask=None):
        if padding_mask is None:
            padding_mask = tokens.eq(self.pad_id)

        causal = causal_attention_mask(tokens.size(1), tokens.device)
        hidden = self.position(self.token_embedding(tokens))

        for depth, layer in enumerate(self.layers):
            hidden = layer(hidden, memory, tgt_mask=causal, tgt_key_padding_mask=padding_mask)
            hidden = self.films[depth](hidden, condition)

        return self.out(self.norm(hidden))

    def forward(self, z, tokens, prot_emb, padding_mask=None):
        memory, condition = self._memory(z, prot_emb)

        return self._decode(memory, tokens, condition, padding_mask=padding_mask)  # [B, S, vocab_size]

    @torch.no_grad()
    def generate(self, z, prot_emb, max_length=None, temperature=1.0, seed=None):
        memory, condition = self._memory(z, prot_emb)

        tokens = autoregressive_sample(
            lambda tok, mem: self._decode(mem, tok, condition)[:, -1],
            n_samples=z.size(0),
            bos_id=self.bos_id, eos_id=self.eos_id, pad_id=self.pad_id,
            max_length=max_length or self.max_length,
            device=z.device, conditions=(memory,),
            temperature=temperature, seed=seed, forbidden_ids=self.forbidden_generation_ids,
            require_ids=() if self.tokenizer is None else self.tokenizer.content_ids)

        return tokens if self.tokenizer is None else self.tokenizer.decode_batch(tokens)

class MultiViewFusion(nn.Module):
    def __init__(self, dim=512, num_heads=8, ff_dim=512, dropout=0.1, num_layers=2, fingerprint_dim=None):
        super().__init__()
        self.fingerprint_proj = nn.Linear(fingerprint_dim or dim, dim)
        self.view_embedding = nn.Embedding(3, dim)  # fingerprint, graph, sequence

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, fp, graph_embs=None, seq_embs=None, graph_mask=None, seq_mask=None):
        B = fp.shape[0]
        fp = self.fingerprint_proj(fp)
        tokens = [fp.unsqueeze(1) + self.view_embedding.weight[0]]
        masks = [torch.zeros(B, 1, dtype=torch.bool, device=fp.device)]

        if graph_embs is not None:
            tokens.append(graph_embs + self.view_embedding.weight[1])
            if graph_mask is None:
                graph_mask = torch.zeros(B, graph_embs.shape[1], dtype=torch.bool, device=fp.device)
            masks.append(graph_mask)

        if seq_embs is not None:
            tokens.append(seq_embs + self.view_embedding.weight[2])
            if seq_mask is None:
                seq_mask = torch.zeros(B, seq_embs.shape[1], dtype=torch.bool, device=fp.device)
            masks.append(seq_mask)

        x = torch.cat(tokens, dim=1)
        padding_mask = torch.cat(masks, dim=1)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        return self.norm(x[:, 0])


class ResidualProteinPrior(nn.Module):
    """Two-layer GELU adapter that starts as the exact identity function."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, protein_embedding):
        correction = self.fc2(self.activation(self.fc1(protein_embedding)))
        return protein_embedding + correction


class SelVAEGen(nn.Module): # so pretty :)
    """Multi-view molecular VAE that gives every target its own region of latent space."""
    def __init__(self, gnn_encoder=None, gnn_decoder=None, seq_encoder=None, seq_decoder=None,
        fusion=None, criterion=None, latent_dim=512, fusion_dim=512, seq_pad_id=0,
        protein_dim=128, prior_hidden=256):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.gnn_decoder = gnn_decoder
        self.seq_encoder = seq_encoder
        self.seq_decoder = seq_decoder

        self.seq_pad_id = seq_pad_id

        self.fusion = fusion
        self.criterion = criterion

        self.to_mu = nn.Linear(fusion_dim, latent_dim)
        self.to_logvar = nn.Linear(fusion_dim, latent_dim)

        self.protein_prior = ResidualProteinPrior(protein_dim, prior_hidden)

    def set_protein_prior_trainable(self, trainable):
        """Freeze or unfreeze the map that defines target centres."""
        for parameter in self.protein_prior.parameters():
            parameter.requires_grad_(trainable)

    def encode(self, data_batch):
        fp = data_batch.fingerprint.float()
        graph_embs, seq_embs = None, None
        graph_mask, seq_mask = None, None

        if self.gnn_encoder is not None:
            graph_embs = self.gnn_encoder(data_batch.x, data_batch.edge_index, data_batch.edge_attr)
            graph_embs, graph_valid = to_dense_batch(graph_embs, data_batch.batch)
            graph_mask = ~graph_valid

        if self.seq_encoder is not None:
            seq_mask = data_batch.drug_tokens.eq(self.seq_pad_id)
            seq_embs = self.seq_encoder(data_batch.drug_tokens, padding_mask=seq_mask)

        h = self.fusion(fp, graph_embs=graph_embs, seq_embs=seq_embs, graph_mask=graph_mask, seq_mask=seq_mask)
        mu, logvar = self.to_mu(h), self.to_logvar(h)

        return mu, logvar

    def prior_mean(self, prot_emb):
        """Centre of the latent region belonging to a target: z ~ N(m(P), I)."""
        return self.protein_prior(prot_emb.float())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, data_batch):
        """Decode z against the batch's target."""
        prot_emb = data_batch.prot_emb
        outputs = {}

        if self.gnn_decoder is not None:
            outputs["graph"] = self.gnn_decoder(z, prot_emb, data_batch=data_batch)

        if self.seq_decoder is not None:
            tokens = data_batch.drug_tokens
            outputs["seq"] = self.seq_decoder(z, tokens, prot_emb, padding_mask=tokens.eq(self.seq_pad_id))

        return outputs

    def compute_loss(self, batch):
        """The shared training contract: (loss, metrics) for one batch."""
        return self.criterion(self, batch)

    @torch.no_grad()
    def generate(self, prot_emb, prot_seq=None, seed=0, temperature=1.0, num_atoms=None):
        """One molecule per row of prot_emb, from that target's region of latent space."""
        was_training = self.training
        self.eval()

        try:
            generator = torch.Generator(device=prot_emb.device).manual_seed(seed)
            z = self.prior_mean(prot_emb) + torch.randn(
                prot_emb.shape[0], self.to_mu.out_features,
                device=prot_emb.device, generator=generator)

            outputs = {}
            if self.gnn_decoder is not None:
                outputs["graph"] = self.gnn_decoder.generate(z, prot_emb, target_atoms=num_atoms, temperature=temperature, seed=seed)
            if self.seq_decoder is not None:
                outputs["seq"] = to_mols(self.seq_decoder.generate(z, prot_emb, temperature=temperature, seed=seed))
        finally:
            self.train(was_training)

        return outputs


def build_selvaegen(tokenizer, protein_dim, max_nodes, max_smiles_len, criterion,
                    gnn_encoder=True, seq_encoder=True, gnn_decoder=True, seq_decoder=True,
                    fusion_dim=512, fingerprint_dim=512):
    """Assemble the requested branches."""
    # ResidualProteinPrior starts as the identity, so the latent matches the embedding width
    latent_dim = protein_dim

    return SelVAEGen(
        gnn_encoder=GNNEncoder(out_channels=fusion_dim) if gnn_encoder else None,
        gnn_decoder=(GNNDecoder(latent_dim=latent_dim, max_nodes=max_nodes,
                                protein_dim=protein_dim) if gnn_decoder else None),
        seq_encoder=(SequenceEncoder(tokenizer.vocab_size, d_model=fusion_dim,
                                    max_length=max_smiles_len, pad_id=tokenizer.pad_id)
                        if seq_encoder else None),
        seq_decoder=(SequenceDecoder(tokenizer.vocab_size, tokenizer=tokenizer,
                                    latent_dim=latent_dim, d_model=fusion_dim,
                                    max_length=max_smiles_len, protein_dim=protein_dim)
                        if seq_decoder else None),
        fusion=MultiViewFusion(dim=fusion_dim, fingerprint_dim=fingerprint_dim),
        criterion=criterion,
        latent_dim=latent_dim, fusion_dim=fusion_dim,
        seq_pad_id=tokenizer.pad_id, protein_dim=protein_dim)
