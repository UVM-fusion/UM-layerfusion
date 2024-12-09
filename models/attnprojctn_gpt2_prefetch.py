from typing import List
import math

import torch
import torch.nn as nn
from torch import ops
from prefetch import Prefetcher

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, n_head, pdrop=0.1):
        super().__init__()

        n_state = nx
        assert n_state % n_head == 0

        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = n_head
        self.split_size = n_state

        # Layer for computing Q, K, V, and the fused projection
        self.c_attn_proj = Conv1D(3 * n_state, nx)  # This will do the Q, K, V + projection in one go
        self.attn_dropout = nn.Dropout(pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)

        # causal mask
        mask = self.bias[:, :, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # merge heads

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, hidden_states):
        # Combine the Q, K, V projection and attention computation in one step
        query, key, value = self.c_attn_proj(hidden_states).split(self.split_size, dim=2)
        query, key, value = self.split_heads(query), self.split_heads(key, k=True), self.split_heads(value)

        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        
        return a


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, num_positions, pdrop=0.1):
        super(Block, self).__init__()

        self.attn = Attention(embed_dim, num_positions, num_heads, pdrop=pdrop)

        # LayerNorm and MLP (no fusion here, but could be done separately if needed)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            Conv1D(embed_dim * 4, embed_dim),
            GELU(),
            Conv1D(embed_dim, embed_dim * 4),
            nn.Dropout(p=pdrop, inplace=False),
        )

    def forward(self, x):
        hidden_states = x
        ln_1_out = self.ln_1(x)
        
        # Attention + Projection Fusion (already fused in the Attention module)
        hidden_states = hidden_states + self.attn(ln_1_out)
        
        ln_2_out = self.ln_2(hidden_states)
        m = self.mlp(ln_2_out)
        hidden_states = hidden_states + m

        return hidden_states


class GPT2(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1600,
        num_heads: int = 25,
        num_layers: int = 48,
        num_positions: int = 1024,
        vocab_size: int = 50257,
        pdrop: float = 0.1,
    ):
        super(GPT2, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_positions = num_positions
        self.vocab_size = vocab_size
        self.pdrop = pdrop

        # Embedding Layers
        self.token_embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
        self.drop = nn.Dropout(self.pdrop)

        # Transformer Layers
        self.layers = nn.Sequential(*[
            Block(self.embed_dim, self.num_heads, self.num_positions, pdrop=self.pdrop)
            for _ in range(self.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # 1. Get embeddings
        h = self.token_embeddings(x.long())
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.drop(h)

        # 2. Pass it through transformer layers
        h = self.layers(h)

        # 3. Apply the last layer norm
        h = self.ln_f(h)
        return h


class PrefetchGPT2LM(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1600,
        num_heads: int = 25,
        num_layers: int = 48,
        num_positions: int = 1024,
        vocab_size: int = 50257,
        pdrop: float = 0.1,
        num_prefetch_streams: int = 2,
    ):
        super(PrefetchGPT2LM, self).__init__()

        self.transformer = GPT2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_positions=num_positions,
            vocab_size=vocab_size,
            pdrop=pdrop
        )

        # Logit Linear Layer
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        h = self.transformer(x)
        logits = self.lm_head(h)
        return logits


