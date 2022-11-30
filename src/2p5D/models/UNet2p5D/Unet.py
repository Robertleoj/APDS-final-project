import torch
import torch.nn as nn

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
    def __init__(self, *, dim, heads, dim_head):
        super().__init__()
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

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, *, dim, heads, dim_head):
        super().__init__()
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


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class ResNetBlock(nn.Module):
    def __init__(self, *, dim, dim_in=None):
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


    def forward(self, x):
        hidden = self.block1(x)
        hidden = self.block2(hidden)

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

class Encoder(nn.Module):
    def __init__(self, *, dims, n_res_blocks, attn_heads, attn_head_dim):
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
                attn = LinearAttention(
                    dim=d, 
                    heads=attn_heads, 
                    dim_head=attn_head_dim
                )

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
    def __init__(self, *, dims, n_res_blocks, attn_heads, attn_head_dim):
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
                attn = LinearAttention(
                    dim=d, 
                    heads=attn_heads,
                    dim_head=attn_head_dim
                )

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
    def __init__(self, *, dim, n_res_blocks, attn_heads, attn_head_dim):
        super().__init__()
        self.backbone_block = BackboneBlock(
            dim=dim, n_res_blocks=n_res_blocks
        )

        # self.attn = Attention(
        self.attn = LinearAttention(
            dim=dim, 
            heads=attn_heads, 
            dim_head=attn_head_dim
        )


    def forward(self, x):
        x = self.backbone_block(x)
        return self.attn(x)
   
class DimensionAdder(nn.Module):
    def __init__(self, *, in_dim, out_dim, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.projection = nn.Conv2d(
            in_dim, 
            n_classes * out_dim, 
            kernel_size=3,
            padding=1
        )

    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        x = rearrange(x, 'b (d c) w h -> b c d w h', c=self.n_classes)
        return x

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

        self.dim_adder = DimensionAdder(
            n_classes=n_classes,
            in_dim=dim,
            out_dim=dim,
        )

    def forward(self, x):
        x, enc_residuals = self.encoder(x)
        x = self.bottom(x)
        x = self.decoder(x, enc_residuals)
        x = self.dim_adder(x)
        return x

        


