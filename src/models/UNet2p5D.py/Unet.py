import torch
import torch.nn as nn


def default(x, y):
    return x if x is not None else y


def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), kernel_size=3, padding = 1)
    )


def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=4, stride=2, padding=1)


class Attention(nn.Module):
    def __init__(self, *, dim, n_heads, head_dim):
        super().__init__()
    

    def forward(self, x):
        pass

class LinearAttention(nn.Module):
    def __init__(self, *, dim, n_heads, head_dim):
        super().__init__()

        pass

    def forward(self, x):
        pass


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

        self.blocks = nn.Sequential(
            *[
                ResNetBlock(dim) 
                if i != 0 or dim_in is None else 
                ResNetBlock(dim_in=dim_in, dim=dim)
                for i in range(n_res_blocks)
            ]
        )

    def forward(self, x):
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

            attn = LinearAttention(dim=d, n_heads=attn_heads, head_dim=attn_head_dim)
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

        dims = reversed(dims)

        for us, d in zip(dims, dims[1:]):

            upsample = Upsample(dim=us, dim_out=d)
            self.upsamples.append(upsample)

            block = BackboneBlock(
                dim=d, dim_in=2 * d, n_res_blocks=n_res_blocks
            )
            
            self.bb_blocks.append(block)

            attn = LinearAttention(dim=d, n_heads=attn_heads, head_dim=attn_head_dim)
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
    def __init__(self, *, stuff):
        super().__init__()
        pass

    def forward(self, x):
        pass
   
class DimensionAdder(nn.Module):
    def __init__(self, *, stuff):
        super().__init__()
        pass


    def forward(self, x):
        pass

class Unet2p5D(nn.Module):
    def __init__(self, *, dim, dim_mults):
        super().__init__()

        dims = [dim * d for d in dim_mults]

        self.encoder = Encoder(dims=dims)
        self.decoder = Decoder(dims=dims)
        self.bottom = UNetBottom()
        self.dim_adder = DimensionAdder()

    def forward(self, x):
        x, enc_residuals = self.encoder(x)
        x = self.bottom(x)
        x = self.decoder(x, enc_residuals)
        x = self.dim_adder(x)
        return x

        


