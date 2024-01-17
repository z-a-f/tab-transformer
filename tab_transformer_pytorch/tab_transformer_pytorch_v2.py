import torch
import torch.nn.functional as F
from torch import nn, einsum

import math
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
# mlp

class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

def positional_encoding(L, embedding_dim):
    pos_enc = torch.zeros((L, embedding_dim))
    for pos in range(L):
        for i in range(0, embedding_dim, 2):
            pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embedding_dim)))
            pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embedding_dim)))
    return pos_enc

# main class

class TabTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        max_text_length,
        num_continuous,
        vocabulary_size,
        vocabulary_dim,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 2,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_shared_categ_embed = True,
        shared_categ_dim_divisor = 8.   # in paper, they reserve dimension / 8 for category shared embedding
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0 + max_text_length, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        self.max_text_length = max_text_length  # This is concatenation of all the text columns, but each truncated

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)

        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)

        # take care of shared category embed

        self.use_shared_categ_embed = use_shared_categ_embed

        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std = 0.02)

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        # textual
        self.vocabulary_dim = vocabulary_dim
        self.text_embed = nn.Embedding(vocabulary_size, vocabulary_dim)
        if vocabulary_dim != dim:
            self.text_embed_reduce = nn.Linear(vocabulary_dim, dim)
        else:
            self.text_embed_reduce = nn.Identity()
        self.pos_encoding = positional_encoding(self.max_text_length, self.vocabulary_dim).unsqueeze(0)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            if exists(continuous_mean_std):
                assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
            self.register_buffer('continuous_mean_std', continuous_mean_std)

            self.norm = nn.LayerNorm(num_continuous)

        # transformer

        self.transformer_categ = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        self.transformer_text = Transformer(
            dim = vocabulary_dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous + (vocabulary_dim * max_text_length)

        hidden_dimensions = [input_size * t for t in  mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act = mlp_act)

    def forward(self, x_categ, x_cont, x_text, return_attn = False):
        xs = []

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            categ_embed = self.category_embed(x_categ)

            if self.use_shared_categ_embed:
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b = categ_embed.shape[0])
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim = -1)

            x, attns_categ = self.transformer_categ(categ_embed, return_attn = True)

        flat_x = rearrange(x, 'b ... -> b (...)')
        xs.append(flat_x)

        assert x_text.shape[-1] == self.max_text_length, f'you must pass in {self.max_text_length} of tokens for your text input'

        if self.max_text_length > 0:
            x_text = self.text_embed(x_text) + self.pos_encoding
            x_text = self.text_embed_reduce(x_text)

            x, attns_text = self.transformer_text(categ_embed, return_attn = True)

            flat_text = rearrange(x, 'b ... -> b (...)')
            xs.append(flat_text)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if self.num_continuous > 0:
            if exists(self.continuous_mean_std):
                mean, std = self.continuous_mean_std.unbind(dim = -1)
                x_cont = (x_cont - mean) / std

            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)

        x = torch.cat(xs, dim = -1)
        logits = self.mlp(x)

        if not return_attn:
            return logits

        return logits, attns_categ, attns_text
