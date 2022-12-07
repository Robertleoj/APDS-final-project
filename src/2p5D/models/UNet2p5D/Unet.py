import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def default(x, y):
    return x if x is not None else y


def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), kernel_size=3, padding = 1)
    )


def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=4, stride=2, padding=1)


class LinearAttention(nn.Module):
    def __init__(self, *, dim, heads, dim_head=None):
        super().__init__()
        if dim_head is None:
            dim_head = dim // heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        # q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, *, dim, heads, dim_head=None):
        super().__init__()

        if dim_head is None:
            dim_head = dim // heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x) + x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialAttentionConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = SpatialAttentionConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=6, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class ResNetBlock(nn.Module):
    def __init__(self, *, dim, dim_in=None, cbam=True):
        super().__init__()

        block = lambda :  nn.Sequential(
            LayerNorm(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

        if dim_in is not None:
            self.block1 = nn.Sequential(
                LayerNorm(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, dim, kernel_size=3, padding=1)
            )
        else:
            self.block1 = block()
        self.block2 = block()

        if cbam:
            self.cbam = CBAM(dim, min(dim//3, 16))

    def forward(self, x):
        hidden = self.block1(x)
        hidden = self.block2(hidden)
        if self.cbam is not None:
            hidden = self.cbam(hidden)

        return hidden + x

class BackboneBlock(nn.Module):
    def __init__(self, *, dim, dim_in=None, n_res_blocks):
        super().__init__()


        if dim_in is not None:

            self.first = nn.Conv2d(
                dim_in, 
                dim, 
                kernel_size=3,
                padding=1
            )

        else:
            self.first = None

        self.blocks = nn.Sequential(
            *[
                ResNetBlock(dim=dim) 
                for i in range(n_res_blocks)
            ]
        )

    def forward(self, x):
        if self.first is not None:
            x = self.first(x)

        return self.blocks(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Encoder(nn.Module):
    def __init__(self, *, dims, n_res_blocks, attn_heads, attn_head_dim=None):
        super().__init__()

        self.bb_blocks = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])


        for d, ds in zip(dims, dims[1:]):

            block = BackboneBlock(dim=d, n_res_blocks=n_res_blocks)
            self.bb_blocks.append(block)

            if d == dims[0]:
                attn = nn.Identity()
            else:
                attn = Residual(PreNorm(d, LinearAttention(
                    dim=d, 
                    heads=attn_heads, 
                    dim_head=attn_head_dim
                )))

            self.attentions.append(attn)

            downsample = Downsample(dim=d, dim_out=ds)
            self.downsamples.append(downsample)


    def forward(self, x):
        res_outputs = []

        for bb, attn, ds in zip(self.bb_blocks, self.attentions, self.downsamples):
            r = bb(x)
            res_outputs.append(r)
            x = attn(r)
            x = ds(x)

        return x, res_outputs


class Decoder(nn.Module):
    def __init__(self, *, dims, n_res_blocks, attn_heads, attn_head_dim=None):
        super().__init__()

        self.upsamples = nn.ModuleList([])
        self.bb_blocks = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        dims = dims[::-1]

        for us, d in zip(dims, dims[1:]):

            upsample = Upsample(dim=us, dim_out=d)
            self.upsamples.append(upsample)

            block = BackboneBlock(
                dim=d, dim_in=2 * d, n_res_blocks=n_res_blocks
            )
            
            self.bb_blocks.append(block)

            if d == dims[-1]:
                attn = nn.Identity()
            else:
                attn = Residual(PreNorm(d, LinearAttention(
                    dim=d, 
                    heads=attn_heads,
                    dim_head=attn_head_dim
                )))

            self.attentions.append(attn)


    def forward(self, x, enc_outputs):
        
        enc_outputs = reversed(enc_outputs)

        for ups, enc_out, bb, attn in zip(
            self.upsamples,
            enc_outputs,
            self.bb_blocks, 
            self.attentions, 
        ):
            x = ups(x)
            x = torch.concat((enc_out, x), dim=1)
            x = bb(x)
            x = attn(x)

        return x


class UNetBottom(nn.Module):
    def __init__(self, *, dim, n_res_blocks, attn_heads, attn_head_dim=None):
        super().__init__()
        self.backbone_block = BackboneBlock(
            dim=dim, n_res_blocks=n_res_blocks
        )

        # self.attn = Attention(
        self.attn = Residual(PreNorm(dim, LinearAttention(
            dim=dim, 
            heads=attn_heads, 
            dim_head=attn_head_dim
        )))


    def forward(self, x):
        x = self.backbone_block(x)
        return self.attn(x)
   
class Unet2p5D(nn.Module):
    def __init__(self, *, 
        dim, 
        n_classes,
        dim_mults,
        attn_heads,
        attn_head_dim,
        n_res_blocks
    ):
        super().__init__()

        dims = [dim * d for d in dim_mults]

        self.encoder = Encoder(
            dims=dims,
            n_res_blocks=n_res_blocks,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim
        )

        self.decoder = Decoder(
            dims=dims,
            n_res_blocks=n_res_blocks,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim
        )

        self.bottom = UNetBottom(
            dim=dims[-1],
            n_res_blocks=n_res_blocks,
            attn_head_dim=attn_head_dim,
            attn_heads=attn_heads
        )

    def forward(self, x):
        # x = (x + 200) / (200 + 200)
        x, enc_residuals = self.encoder(x)
        x = self.bottom(x)
        x = self.decoder(x, enc_residuals)
        return x

        


