from dataclasses import dataclass

from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ....geometry.projection import sample_image_grid
from ...encodings.positional_encoding import PositionalEncoding
from ...transformer.transformer import Transformer


@dataclass
class ImageSelfAttentionCfg:
    patch_size: int
    num_octaves: int
    num_layers: int
    num_heads: int
    d_token: int
    d_dot: int
    d_mlp: int


class ImageSelfAttention(nn.Module):
    positional_encoding: nn.Sequential
    patch_embedder: nn.Sequential
    transformer: Transformer

    def __init__(
        self,
        cfg: ImageSelfAttentionCfg,
        d_in: int,
        d_out: int,
    ):
        super().__init__()
        self.positional_encoding = nn.Sequential(
            (pe := PositionalEncoding(cfg.num_octaves)),
            nn.Linear(pe.d_out(2), cfg.d_token),
        )
        self.patch_embedder = nn.Sequential(
            nn.Conv2d(d_in, cfg.d_token, cfg.patch_size, cfg.patch_size),
            nn.ReLU(),
        )
        self.transformer = Transformer(
            cfg.d_token,
            cfg.num_layers,
            cfg.num_heads,
            cfg.d_dot,
            cfg.d_mlp,
        )
        self.resampler = nn.ConvTranspose2d(
            cfg.d_token,
            d_out,
            cfg.patch_size,
            cfg.patch_size,
        )
        self.index = 0
    def forward(
        self,
        image: Float[Tensor, "batch d_in height width"],
    ) -> Float[Tensor, "batch d_out height width"]:
        # Embed patches so they become tokens.
        tokens = self.patch_embedder.forward(image)

        # Append positional information to the tokens.
        _, _, nh, nw = tokens.shape
        if nh <20:
            index= self.index//4   #查看是第几个crop
            self.index=self.index+1
            i=index//2
            j=index%2   #行和列的缩影
            xy, _ = sample_image_grid((nh*2, nw*2), device=image.device)
            xy = self.positional_encoding.forward(xy)[i*nh:(i+1)*nh,j*nw :(j+1)*nw, :]
        else:  #走nograd全图将index赋值为0
            self.index=0
            xy, _ = sample_image_grid((nh, nw), device=image.device)
            xy = self.positional_encoding.forward(xy)    
        tokens = tokens + rearrange(xy, "nh nw c -> c nh nw")

        # Put the tokens through a transformer.
        _, _, nh, nw = tokens.shape

        if nh>=20 :
            for i in range(2):
                for j in range(2):    
                    tokens_1=tokens[:,:,i*nh//2:(i+1)*nh//2,j*nw//2:(j+1)*nw//2]  
                    tokens_1 = rearrange(tokens_1, "b c nh nw -> b (nh nw) c")
                    tokens_1 = self.transformer.forward(tokens_1)
                    tokens_1 = rearrange(tokens_1, "b (nh nw) c -> b c nh nw", nh=nh//2, nw=nw//2)
                    tokens[:,:,i*nh//2:(i+1)*nh//2,j*nw//2:(j+1)*nw//2]=tokens_1
        else:
            tokens = rearrange(tokens, "b c nh nw -> b (nh nw) c")
            tokens = self.transformer.forward(tokens)
            tokens = rearrange(tokens, "b (nh nw) c -> b c nh nw", nh=nh, nw=nw)

        tokens = self.resampler.forward(tokens)

        return tokens
