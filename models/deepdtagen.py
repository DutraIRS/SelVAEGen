# DeepDTAGen (Shah et al., Nature Communications 16, 5021, 2025)
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.utils import to_dense_batch

from utils.chemical_tools import ATOM_NUM_FEATURES, to_mols
from utils.data_tools import MAX_SEQ_LEN

from .components import SinusoidalPositionalEncoding, autoregressive_sample, causal_attention_mask, forbidden_generation_ids


class GatedProteinCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_filters=32, output_dim=128,
                protein_length=MAX_SEQ_LEN, kernel_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        widths = [protein_length, n_filters, n_filters * 2, n_filters * 3]
        self.values = nn.ModuleList(nn.Conv1d(widths[i], widths[i + 1], kernel_size) for i in range(3))
        self.gates = nn.ModuleList(nn.Conv1d(widths[i], widths[i + 1], kernel_size) for i in range(3))

        conv_width = embed_dim - 3 * (kernel_size - 1)
        self.flat_dim = widths[-1] * conv_width
        self.fc = nn.Linear(self.flat_dim, output_dim)

    def forward(self, sequence):
        x = self.embedding(sequence)
        for value, gate in zip(self.values, self.gates):
            x = F.relu(value(x) * torch.sigmoid(gate(x)))
        return self.fc(x.flatten(1)), x


class GraphEncoder(nn.Module):
    def __init__(self, num_features, condition_dim, output_dim=128, dropout=0.2):
        super().__init__()
        d_model = num_features * 4
        self.conv1 = GCNConv(num_features, num_features * 2)
        self.conv2 = GCNConv(num_features * 2, num_features * 3)
        self.conv3 = GCNConv(num_features * 3, num_features * 4)

        self.to_mu = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.ReLU(),
                                nn.Linear(d_model, d_model))
        self.to_logvar = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.ReLU(),
                                    nn.Linear(d_model, d_model))

        self.cond = nn.Linear(condition_dim, d_model)

        self.node_seg_encoding = nn.Parameter(torch.randn(d_model))

        self.pooled_head = nn.Sequential(
            nn.Linear(num_features * 4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim))

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        nodes, real = to_dense_batch(x, batch)      # [batch, max_nodes, d_model], [batch, max_nodes]
        nodes = nodes + self.node_seg_encoding

        return nodes, ~real, self.pooled_head(global_max_pool(x, batch))

    def sample(self, nodes, real, protein_map, affinity):
        mu = self.to_mu(nodes)
        logvar = -self.to_logvar(nodes).abs()

        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        condition = self.cond(protein_map.flatten(1)).unsqueeze(1)
        z = z + condition + affinity.view(-1, 1, 1)

        mask = real.unsqueeze(-1).float()
        kl = -0.5 * ((1 + logvar - mu.pow(2) - logvar.exp()) * mask).sum() / mask.sum().clamp(min=1)

        return z, kl


class DeepDTAGen(nn.Module):
    """Multitask affinity prediction and target-aware generation."""
    def __init__(self, tokenizer, num_features=ATOM_NUM_FEATURES, embed_dim=128, n_filters=32,
                    output_dim=128, protein_vocab_size=26, protein_length=MAX_SEQ_LEN,
                    nhead=8, decoder_layers=8, fusion_layers=8, dim_feedforward=1024,
                    dropout=0.1, encoder_dropout=0.2, head_dropout=0.3, max_length=128,
                    w_recon=1.0, w_affinity=2.0, w_kl=0.001, stochastic_nodes=48,
                    generation_affinity=6.0):
        super().__init__()

        d_model = num_features * 4

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_features = num_features
        self.stochastic_nodes = stochastic_nodes
        self.generation_affinity = generation_affinity
        self.w_recon, self.w_affinity, self.w_kl = w_recon, w_affinity, w_kl

        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.forbidden_ids = forbidden_generation_ids(tokenizer)

        self.protein_encoder = GatedProteinCNN(protein_vocab_size, embed_dim, n_filters,
                                            output_dim, protein_length, kernel_size=8)
        self.graph_encoder = GraphEncoder(num_features,
                                            condition_dim=self.protein_encoder.flat_dim,
                                            output_dim=output_dim, dropout=encoder_dropout)

        self.latent_seg_encoding = nn.Parameter(torch.randn(d_model))

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True, norm_first=True)
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, fusion_layers,
                                                    norm=nn.LayerNorm(d_model))

        self.affinity_head = nn.Sequential(
            nn.Linear(2 * output_dim, 1024), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(256, 1),
        )

        self.smiles_embedding = nn.Embedding(tokenizer.vocab_size, d_model,
                                                padding_idx=self.pad_id)
        self.position = SinusoidalPositionalEncoding(d_model, max_length, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers,
                                            norm=nn.LayerNorm(d_model))

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model), nn.PReLU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, tokenizer.vocab_size))
        nn.init.zeros_(self.out[3].bias)

        self.recon_loss = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.affinity_loss = nn.MSELoss()

    def fuse(self, nodes, node_padding, z):
        latent = z + self.latent_seg_encoding
        memory = torch.cat([nodes, latent], dim=1)

        # the latent is per node, so unmasked padding would make the output depend on the
        # largest graph in the batch
        memory_padding = torch.cat([node_padding, node_padding], dim=1)

        return self.fusion_encoder(memory, src_key_padding_mask=memory_padding), memory_padding

    @staticmethod
    def normalise(x):
        """Row-normalised atom features, as create_data.py builds them."""
        # GCNConv has no input norm, so raw 0/1 rows shift the latent head's scale
        return x / x.sum(-1, keepdim=True).clamp_min(1e-6)

    def encode(self, data):
        protein_vector, protein_map = self.protein_encoder(data.prot_seq_cat)
        nodes, node_padding, graph_vector = self.graph_encoder(self.normalise(data.x), data.edge_index, data.batch)

        z, kl = self.graph_encoder.sample(nodes, ~node_padding, protein_map, data.affinity.float())

        memory, memory_padding = self.fuse(nodes, node_padding, z)

        return memory, memory_padding, graph_vector, protein_vector, kl

    def decode(self, memory, memory_padding, drug_tokens):
        hidden = self.position(self.smiles_embedding(drug_tokens))
        hidden = self.decoder(
            hidden, memory,
            tgt_mask=causal_attention_mask(drug_tokens.size(1), drug_tokens.device),
            tgt_key_padding_mask=drug_tokens.eq(self.pad_id),
            memory_key_padding_mask=memory_padding,
        )
        return self.out(hidden)

    def compute_loss(self, batch):
        tokens = batch.drug_tokens

        memory, memory_padding, graph_vector, protein_vector, kl = self.encode(batch)

        predicted = self.affinity_head(torch.cat([graph_vector, protein_vector], dim=-1)).squeeze(-1)

        logits = self.decode(memory, memory_padding, tokens[:, :-1])

        recon = self.recon_loss(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
        affinity = self.affinity_loss(predicted, batch.affinity.float())

        total = self.w_recon * recon + self.w_affinity * affinity + self.w_kl * kl

        return total, {'loss': total.item(), 'recon': recon.item(), 'affinity': affinity.item(), 'kl': kl.item()}

    def _sample_tokens(self, memory, memory_padding, seed, temperature, greedy):
        return autoregressive_sample(
            lambda tokens, mem, pad: self.decode(mem, pad, tokens)[:, -1],
            n_samples=memory.size(0),
            bos_id=self.bos_id, eos_id=self.eos_id, pad_id=self.pad_id,
            max_length=self.max_length, device=memory.device,
            conditions=(memory, memory_padding), temperature=temperature, seed=seed,
            forbidden_ids=self.forbidden_ids, require_ids=self.tokenizer.content_ids,
            greedy=greedy)

    @torch.no_grad()
    def generate(self, prot_emb, prot_seq=None, seed=0, temperature=1.0, affinity=None,
                    greedy=False, num_atoms=None):
        """Stochastic mode: the target sequence is the only input, the graph is noise."""
        # the paper reports two strategies, this is the one that needs no seed molecule
        training = self.training
        self.eval()                           # dropout off, or the seed does not hold
        try:
            batch, device = prot_seq.size(0), prot_seq.device
            d_model = self.latent_seg_encoding.numel()
            affinity = self.generation_affinity if affinity is None else affinity

            generator = torch.Generator(device=device).manual_seed(seed)
            shape = (batch, self.stochastic_nodes, d_model)
            # GraphEncoder.forward marks the node half, so the fused memory has to match
            nodes = torch.randn(shape, device=device, generator=generator) + self.graph_encoder.node_seg_encoding
            z = torch.randn(shape, device=device, generator=generator)

            _, protein_map = self.protein_encoder(prot_seq)
            condition = self.graph_encoder.cond(protein_map.flatten(1)).unsqueeze(1)
            z = z + condition + torch.full((batch, 1, 1), affinity, device=device)

            padding = torch.zeros(batch, self.stochastic_nodes, dtype=torch.bool, device=device)
            memory, memory_padding = self.fuse(nodes, padding, z)

            tokens = self._sample_tokens(memory, memory_padding, seed, temperature, greedy)
        finally:
            self.train(training)

        return {'seq': to_mols(self.tokenizer.decode_batch(tokens))}

