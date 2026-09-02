import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_length).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[:encoding[:, 1::2].shape[1]])

        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.encoding[:, :x.size(1)])


class FiLM(nn.Module):
    """Scale and shift a hidden state by a condition, keeping it present between layers."""
    def __init__(self, condition_dim, dim):
        super().__init__()
        # zero-initialised, so it is the identity on the first epoch
        self.to_scale_shift = nn.Linear(condition_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x, condition):
        scale, shift = self.to_scale_shift(condition).chunk(2, dim=-1)

        # broadcast a per-molecule condition over nodes or tokens
        while scale.dim() < x.dim():
            scale, shift = scale.unsqueeze(-2), shift.unsqueeze(-2)

        return x * (1 + scale) + shift


def forbidden_generation_ids(tokenizer):
    """Special tokens a sampler must never emit."""
    return (tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id,
            tokenizer.mask_id, tokenizer.enc_id)


def causal_attention_mask(length, device=None):
    """True where attention is forbidden, i.e., strictly above the diagonal."""
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


@torch.no_grad()
def autoregressive_sample(next_logits, n_samples, *, bos_id, eos_id, pad_id, max_length, device,
                        conditions=(), temperature=1.0, seed=None, forbidden_ids=(),
                        require_ids=(), greedy=False):
    """Sample token sequences autoregressively, padding each row once it emits EOS."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    forbidden_ids = tuple(dict.fromkeys(int(i) for i in forbidden_ids if i is not None and int(i) != eos_id))
    required = (torch.tensor(sorted({int(i) for i in require_ids}), device=device)
                if len(require_ids) else None)

    tokens = torch.full((n_samples, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)
    has_content = torch.zeros(n_samples, dtype=torch.bool, device=device)

    for step in range(max_length - 1):
        logits = next_logits(tokens, *conditions).float()
        if forbidden_ids:
            logits[:, list(forbidden_ids)] = -torch.inf
        if required is not None:
            logits[~has_content, eos_id] = -torch.inf

        if step == 0 and not torch.isfinite(logits).any(dim=-1).all():
            raise RuntimeError("sampling mask left a row with no allowed token, check forbidden_ids/require_ids against the vocabulary")

        if greedy or temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probabilities, num_samples=1, generator=generator)

        if required is not None:
            has_content |= torch.isin(nxt.squeeze(1), required)

        nxt[finished] = pad_id
        tokens = torch.cat([tokens, nxt], dim=1)

        finished |= nxt.squeeze(1).eq(eos_id)
        if bool(finished.all()):
            break

    return tokens
