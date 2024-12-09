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
    def __init__(self, prefetcher, nx, n_ctx, n_head, pdrop=0.1):
        super().__init__()
        self.prefetcher = prefetcher
        n_state = nx
        assert n_state % n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx))
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = n_head
        self.split_size = n_state
        self.c_attn = Conv1D(3 * n_state, nx).cuda()
        self.c_proj = Conv1D(n_state, nx).cuda()
        self.attn_dropout = nn.Dropout(pdrop)
        self.resid_dropout = nn.Dropout(pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd: ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query, key, value = self.split_heads(query), self.split_heads(key, k=True), self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class Block(nn.Module):
    def __init__(self, prefetcher, embed_dim, num_heads, num_positions, pdrop=0.1):
        super(Block, self).__init__()
        self.prefetcher = prefetcher
        self.attn = Attention(prefetcher, embed_dim, num_positions, num_heads, pdrop=pdrop).cuda()

        # Fusion of LayerNorm and MLP
        self.ln_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim).cuda(),
            Conv1D(embed_dim * 4, embed_dim),
            GELU(),
            Conv1D(embed_dim, embed_dim * 4),
            nn.Dropout(p=pdrop, inplace=False)
        ).cuda()

    def forward(self, x):
        # Apply fused LayerNorm + MLP
        hidden_states = self.ln_mlp(x)
        hidden_states = hidden_states + self.attn(x)
        return hidden_states


class GPT2(nn.Module):
    def __init__(
        self,
        prefetcher,
        embed_dim: int = 1600,
        num_heads: int = 25,
        num_layers: int = 48,
        num_positions: int = 1024,
        vocab_size: int = 50257,
        pdrop: float = 0.1,
    ):
        super().__init__()

        self.prefetcher = prefetcher
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_positions = num_positions
        self.vocab_size = vocab_size
        self.pdrop = pdrop

        # Embedding Layers
        self.token_embeddings = nn.Embedding(self.vocab_size, self.embed_dim).cuda()
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim).cuda()
        self.drop = nn.Dropout(self.pdrop).cuda()

        # Transformer Layers
        self.layers = nn.Sequential(*[
            Block(self.prefetcher, self.embed_dim, self.num_heads, self.num_positions, pdrop=self.pdrop)
            for _ in range(self.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(self.embed_dim).cuda()

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
        super().__init__()
        self.prefetcher = Prefetcher(num_prefetch_streams=num_prefetch_streams)

        self.transformer = GPT2(
            prefetcher=self.prefetcher,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_positions=num_positions,
            vocab_size=vocab_size,
            pdrop=pdrop
        )

        # Logit Linear Layer
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False).cuda()

        for module in self.modules():
            self._init_weight(module)

    def forward(self, x):
        h = self.transformer(x)
        logits = self.lm_head(h)
        return logits

    def _init_weight(self, module):
        initializer_range = 0.02  # Default value of Huggingface GPT-2 Config
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

