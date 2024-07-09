from torch import nn
from pydantic import BaseModel


class Config(BaseModel):
    block_size: int = 256
    voacb_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.voacb_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.voacb_size, bias=False)
