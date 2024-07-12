import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).to("cuda")

class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate, num_ids):
        super(Attention2D, self).__init__()
        self.dim = dim
        self.embeds = nn.Embedding(num_ids, dim*dim)
        self.new_embeds = nn.Parameter(torch.randn(dim*dim), requires_grad=True)
        # self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.strength_embedding = _get_sinusoid_encoding_table(3, dim*dim)
        

    def forward(self, q, k, pos, mask=None, embed_id1=None):
        # if embed_id1.item() != 0:
        #     # strength_id = (embed_id1.item() - 1) % 3
        #     # strength_emb = self.strength_embedding[strength_id]
        #     embed = self.embeds(torch.LongTensor([(embed_id1.item()-1)//3 + 1]).to("cuda"))
        #     embed = embed# + strength_emb
        # else:
        #     embed = self.embeds(torch.LongTensor([0]).to("cuda")).detach()
        embed = self.new_embeds
        embed = embed.view(self.dim, self.dim)
        q = F.linear(q, embed)

        k = self.k_fc(k)
        v = self.v_fc(k)
        
        # embed = embed[:, None, :]
        # q = q + embed

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate, num_ids):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention2D(dim, attn_dp_rate, num_ids=num_ids)

    def forward(self, q, k, pos, mask=None, embed_id1=None):
        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask, embed_id1=embed_id1)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None, num_ids=None):
        super(Attention, self).__init__()
        if attn_mode in ["qk", "gate"]:
            self.q_fc = nn.Linear(dim, dim, bias=False)
            self.k_fc = nn.Linear(dim, dim, bias=False)
        if attn_mode in ["pos", "gate"]:
            self.pos_fc = nn.Sequential(
                nn.Linear(pos_dim, pos_dim), nn.ReLU(), nn.Linear(pos_dim, dim // 8)
            )
            self.head_fc = nn.Linear(dim // 8, n_heads)
        if attn_mode == "gate":
            self.gate = nn.Parameter(torch.ones(n_heads))
        # self.v_fc = nn.Linear(dim, dim, bias=False)
        self.dim = dim
        self.embeds = nn.Embedding(num_ids, dim*dim)
        self.new_embeds = nn.Parameter(torch.randn(dim*dim), requires_grad=True)
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.n_heads = n_heads
        self.attn_mode = attn_mode
        self.strength_embedding = _get_sinusoid_encoding_table(3, dim*dim)

    def forward(self, x, pos=None, ret_attn=False, embed_id2=None):
        # if embed_id2.item() != 0:
        #     strength_id = (embed_id2.item() - 1) % 3
        #     strength_emb = self.strength_embedding[strength_id]
        #     embed = self.embeds(torch.LongTensor([(embed_id2.item()-1)//3 + 1]).to("cuda"))
        #     embed = embed + strength_emb
        # else:
        #     embed = self.embeds(torch.LongTensor([0]).to("cuda")).detach()

        if self.attn_mode in ["qk", "gate"]:
            q = self.q_fc(x)
            q = q.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
            k = self.k_fc(x)
            k = k.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        embed = self.new_embeds
        embed = embed.view(self.dim, self.dim)
        v = F.linear(x, embed)
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
        out = out.view(x.shape[0], x.shape[1], -1)
        out = self.dp(self.out_fc(out))
        if ret_attn:
            return out, attn
        else:
            return out


class Transformer(nn.Module):
    def __init__(
        self, dim, ff_hid_dim, ff_dp_rate, n_heads, attn_dp_rate, attn_mode="qk", pos_dim=None, num_ids=None):
        super(Transformer, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention(dim, n_heads, attn_dp_rate, attn_mode, pos_dim, num_ids=num_ids)

    def forward(self, x, pos=None, ret_attn=False, embed_id2=None):
        residue = x
        x = self.attn_norm(x)
        x = self.attn(x, pos, ret_attn, embed_id2=embed_id2)
        if ret_attn:
            x, attn = x
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        if ret_attn:
            return x, attn.mean(dim=1)[:, 0]
        else:
            return x


class TransIBRNet(nn.Module):
    def __init__(self, args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3, ret_alpha=False):
        super(TransIBRNet, self).__init__()
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch + 3, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        )
        self.view_selftrans = nn.ModuleList([])
        self.view_crosstrans = nn.ModuleList([])
        self.q_fcs = nn.ModuleList([])
        for i in range(args.viewtrans_depth):
            view_crosstrans = Transformer2D(
                dim=args.netwidth, ff_hid_dim=int(args.netwidth * 4), ff_dp_rate=0.1, attn_dp_rate=0.1, num_ids=6,
            )
            self.view_crosstrans.append(view_crosstrans)
            view_selftrans = Transformer(
                dim=args.netwidth, ff_hid_dim=int(args.netwidth * 4), n_heads=4, ff_dp_rate=0.1, attn_dp_rate=0.1, num_ids=6,
            )
            self.view_selftrans.append(view_selftrans)
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(args.netwidth + posenc_dim + viewenc_dim, args.netwidth),
                    nn.ReLU(),
                    nn.Linear(args.netwidth, args.netwidth),
                )
            else:
                q_fc = nn.Identity()
            self.q_fcs.append(q_fc)
        
        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim
        self.ret_alpha = ret_alpha
        self.posenc_mode = args.posenc_mode
        self.norm = nn.LayerNorm(args.netwidth)
        self.rgb_fc = nn.Linear(args.netwidth, 3)
        self.relu = nn.ReLU()
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d, embed_id1, embed_id2):
        embed_id1 = embed_id1[0]
        embed_id2 = embed_id2[0]
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = self.view_enc(viewdirs)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)
        embed = torch.cat([pts_, viewdirs_], dim=-1)

        rgb_feat = self.rgbfeat_fc(rgb_feat)
        q = rgb_feat.max(dim=2)[0]
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)
        for i, (crosstrans, q_fc, selftrans) in enumerate(zip(self.view_crosstrans, self.q_fcs, self.view_selftrans)):
            q = crosstrans(q, rgb_feat, ray_diff, mask, embed_id1=embed_id2)
            if i % 2 == 0:
                q = torch.cat((q, input_pts, input_views), dim=-1)
                q = q_fc(q)
            q = selftrans(q, ret_attn=self.ret_alpha, embed_id2=embed_id1)
            if self.ret_alpha:
                q, attn = q
        h = self.norm(q)
        outputs = self.rgb_fc(h.mean(dim=1))
        if self.ret_alpha:
            if self.posenc_mode == "pre_add":
                attn = attn
            return torch.cat([outputs, attn], dim=1)
        else:
            return outputs