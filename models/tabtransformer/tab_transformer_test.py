'''
Pytorch implementation of TabTransformer taken from https://github.com/lucidrains/tab-transformer-pytorch/tree/main
'''

import torch
import torch.nn.functional as F
from torch import nn, einsum
from pytorch_model_summary import summary

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out), attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout)),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = x + attn_out
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# main class

class TabTransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))
        self.ff = PreNorm(dim, FeedForward(dim, dropout=ff_dropout))

    def forward(self, x):
        attn_out, _ = self.attn(x)
        x = x + attn_out
        x = x + self.ff(x)
        return x

class TabTransformer(nn.Module):
    def __init__(self, *, categories, num_continuous, dim, depth, heads, dim_head=16,
                 num_special_tokens=2, continuous_mean_std=None, attn_dropout=0., ff_dropout=0.,
                 use_shared_categ_embed=True, shared_categ_dim_divisor=8.):
        super().__init__()
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.num_special_tokens = num_special_tokens
        self.dim = dim

        total_tokens = sum(categories) + num_special_tokens
        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)
        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)

        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std=0.02)

        if len(categories) > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        if num_continuous > 0:
            if continuous_mean_std is not None:
                assert continuous_mean_std.shape == (num_continuous, 2), \
                    'continuous_mean_std must have a shape of (num_continuous, 2)'
            self.register_buffer('continuous_mean_std', continuous_mean_std)
            self.norm = nn.LayerNorm(num_continuous)

        self.layers = nn.ModuleList([
            TabTransformerLayer(dim, heads, dim_head, attn_dropout, ff_dropout) for _ in range(depth)
        ])

    def forward(self, x_categ, x_cont):
        assert x_categ.shape[-1] == self.num_categories, f'Expected {self.num_categories} category inputs'
        if self.num_categories > 0:
            x_categ = x_categ + self.categories_offset
            categ_embed = self.category_embed(x_categ)

            if hasattr(self, 'shared_category_embed'):
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b=categ_embed.shape[0])
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim=-1)
        else:
            categ_embed = torch.empty((x_categ.shape[0], self.num_categories, self.dim), device=x_categ.device)

        if self.num_continuous > 0:
            if self.continuous_mean_std is not None:
                mean, std = self.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std
            x_cont = self.norm(x_cont)

        return categ_embed, x_cont

    def forward_layer(self, x, layer_idx):
        return self.layers[layer_idx](x)

def main():
    cont_mean_std = torch.randn(10, 2)

    model = TabTransformer(
        categories=(10, 5, 6, 5, 8),  # tuple containing the number of unique values within each category
        num_continuous=10,  # number of continuous values
        dim=32,  # dimension, paper set at 32
        depth=4,  # depth, paper recommended 6
        heads=8,  # heads, paper recommends 8
        attn_dropout=0.1,  # post-attention dropout
        ff_dropout=0.1,  # feed forward dropout
        continuous_mean_std=cont_mean_std  # (optional) - normalize the continuous values before layer norm
    )

    x_categ = torch.randint(0, 5, (1, 5))  # category values, from 0 - max number of categories, in the order as passed into the constructor above
    x_cont = torch.randn(1, 10)  # assume continuous values are already normalized individually

    concatenated_output = model(x_categ, x_cont)  # (1, 1)

    # print(model, (x_categ,x_cont))
    # print(concatenated_output)

if __name__ == "__main__":
    main()