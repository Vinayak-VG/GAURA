## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time
import numpy as np
import math

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()

#         hidden_features = int(dim*ffn_expansion_factor)

#         self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

#         self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x

class FeedForward_Emb(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward_Emb, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
#         self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

#     def forward(self, x, k):
#         b,c,h,w = x.shape

#         kv = self.kv_dwconv(self.kv(k))
#         k,v = kv.chunk(2, dim=1)   
#         q = self.q_dwconv(self.q(x))

#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)

#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         out = self.project_out(out)
#         return out
    
class Attention_Emb(nn.Module):
    def __init__(self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None, num_ids=None):
        super(Attention_Emb, self).__init__()
        if attn_mode in ["qk", "gate"]:
            self.q_embeds = nn.Embedding(num_ids, (dim*dim))
            self.k_embeds = nn.Embedding(num_ids, (dim*dim))
        if attn_mode in ["pos", "gate"]:
            self.pos_fc = nn.Sequential(
                nn.Linear(pos_dim, pos_dim), nn.ReLU(), nn.Linear(pos_dim, dim // 8)
            )
            self.head_fc = nn.Linear(dim // 8, n_heads)
        if attn_mode == "gate":
            self.gate = nn.Parameter(torch.ones(n_heads))
        # self.v_fc = nn.Linear(dim, dim, bias=False)
        self.dim = dim
        self.v_embeds = nn.Embedding(num_ids, (dim*dim))
        q_stdv = 1. / math.sqrt(self.q_embeds.weight.size(1))
        k_stdv = 1. / math.sqrt(self.k_embeds.weight.size(1))
        v_stdv = 1. / math.sqrt(self.v_embeds.weight.size(1))
        self.q_embeds.weight.data.uniform_(-q_stdv, q_stdv)
        self.k_embeds.weight.data.uniform_(-k_stdv, k_stdv)
        self.v_embeds.weight.data.uniform_(-v_stdv, v_stdv)

        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.n_heads = n_heads
        self.attn_mode = attn_mode
        self.strength_linear = nn.Linear(512, dim)

    def forward(self, x, pos=None, ret_attn=False, embed_id1=None, strength=None):
        strength = self.strength_linear(strength)
        x = x.permute(0, 2, 3, 1)
        q_embed = self.q_embeds(torch.LongTensor([embed_id1.item()]).to("cuda"))
        k_embed = self.k_embeds(torch.LongTensor([embed_id1.item()]).to("cuda"))
        v_embed = self.v_embeds(torch.LongTensor([embed_id1.item()]).to("cuda"))

        q_embed = q_embed.view(self.dim, self.dim)
        k_embed = k_embed.view(self.dim, self.dim)
        v_embed = v_embed.view(self.dim, self.dim)        

        if self.attn_mode in ["qk", "gate"]:
            q = F.linear(x, q_embed)
            q = q + strength.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            q = q.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
            k = F.linear(x, k_embed)
            k = k + strength.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            k = k.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        v = F.linear(x, v_embed)
        v = v + strength.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        v = v.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        if self.attn_mode in ["qk", "gate"]:
            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            attn = torch.softmax(attn, dim=-1)
        elif self.attn_mode == "pos":
            pos = self.pos_fc(pos)
            attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            attn = torch.softmax(attn, dim=-1)
        if self.attn_mode == "gate":
            pos = self.pos_fc(pos)
            pos_attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            pos_attn = torch.softmax(pos_attn, dim=-1)
            gate = self.gate.view(1, -1, 1, 1)
            attn = (1.0 - torch.sigmoid(gate)) * attn + torch.sigmoid(gate) * pos_attn
            attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.dp(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(x.shape[0], x.shape[1], x.shape[2], -1)
        out = self.dp(self.out_fc(out))
        if ret_attn:
            return out, attn
        else:
            return out

class Transformer(nn.Module):
    def __init__(
        self, dim, ff_hid_dim, ff_dp_rate, n_heads, attn_dp_rate, attn_mode="qk", pos_dim=None, num_ids=None, 
    ):
        super(Transformer, self).__init__()
        self.attn_norm = LayerNorm(dim, "WithBias")
        self.ff_norm = LayerNorm(dim, "WithBias")
        self.dim = dim
        self.ff = FeedForward_Emb(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention_Emb(dim, n_heads, attn_dp_rate, attn_mode, pos_dim, num_ids=num_ids)

    def forward(self, x, pos=None, ret_attn=False, embed_id1=None, strength=None):
        residue = x

        x = self.attn_norm(x)
        x = self.attn(x, pos, ret_attn, embed_id1=embed_id1, strength=strength)
        x = x.permute(0, 3, 1, 2)
        if ret_attn:
            x, attn = x

        x = x + residue
        residue = x
        x = self.ff_norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ff(x)
        x = x.permute(0, 3, 1, 2)
        x = x + residue

        if ret_attn:
            return x, attn.mean(dim=1)[:, 0]
        else:
            return x



## Transformer Block
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
#         super(TransformerBlock, self).__init__()

#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.norm1k = LayerNorm(dim, LayerNorm_type)
#         self.attn = Attention(dim, num_heads, bias)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

#     def forward(self, x, k):
#         x = x + self.attn(self.norm1(x), self.norm1k(k))
#         x = x + self.ffn(self.norm2(x))

#         return x


##########################################################################
##---------- Prompt Gen Module -----------------------
# class PromptGenBlock(nn.Module):
#     def __init__(self, prompt_dim=128, lin_dim = 192, num_ids = 5):
#         super(PromptGenBlock,self).__init__()
#         self.prompt_param = nn.Embedding(num_ids, lin_dim)
#         # self.new_prompt_param = nn.Parameter(torch.randn(lin_dim), requires_grad=True)
#         # self.linear_layer = nn.Linear(lin_dim, lin_dim)
#         self.linear_layer = nn.Conv2d(lin_dim, lin_dim, kernel_size=1, stride=1, padding=0, bias=False)
#         self.conv3x3 = nn.Conv2d(lin_dim, lin_dim, kernel_size=3, stride=1, padding=1, bias=False)
#         self.prompt_dim = lin_dim
#         self.transform_embed = nn.Linear(256, lin_dim)

#     def forward(self, x, embed_id):
#         B, C, H, W = x.shape
#         # x = x.mean(dim=(-2,-1))
#         prompt_weights = F.sigmoid(self.linear_layer(x))

#         embedding = self.prompt_param(torch.LongTensor([embed_id.item()]).to("cuda"))
#         prompt = prompt_weights * embedding.view(self.prompt_dim).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat(B,1,1,1)
        
#         prompt = F.interpolate(prompt, (H,W), mode="bilinear")
#         prompt = self.conv3x3(prompt)

        # return prompt

class PromptBlock(nn.Module):
    def __init__(self, bias = False, prompt_dim = 128, lin_dim = 192, num_ids = 5):
        super(PromptBlock,self).__init__()
        # self.prompt1 = PromptGenBlock(prompt_dim = prompt_dim, lin_dim = lin_dim, num_ids = num_ids)
        # self.noise_level1 = TransformerBlock(dim = lin_dim, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.noise_level1 = Transformer(dim=lin_dim, ff_hid_dim=int(lin_dim * 4), n_heads=4, ff_dp_rate=0.1, attn_dp_rate=0.1, num_ids=num_ids)
        self.reduce_noise_level1 = nn.Conv2d(lin_dim, lin_dim, kernel_size=1, bias=bias)
        self.noise_level2 = Transformer(dim=lin_dim, ff_hid_dim=int(lin_dim * 4), n_heads=4, ff_dp_rate=0.1, attn_dp_rate=0.1, num_ids=num_ids)
        self.reduce_noise_level2 = nn.Conv2d(lin_dim, lin_dim, kernel_size=1, bias=bias)

    def forward(self, x, embed_id, strength):
        x = self.reduce_noise_level1(self.noise_level1(x, embed_id1=embed_id, strength=strength))
        x = self.reduce_noise_level2(self.noise_level2(x, embed_id1=embed_id, strength=strength))
        return x