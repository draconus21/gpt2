import math
import torch
import tiktoken
from torch import nn
from torch.nn import functional as F
from pydantic import BaseModel

from dataloader import DataLoaderLite


class Config(BaseModel):
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        # projections for q, k, and v
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)  # (B, T, C*3)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # apply multi-head attention
        # bring the head dim forward to be a batch dim (pytorch likes this)
        # nh: number of heads, hs: head size, c: number of channels (hs*ns)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # compute attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        # mask the future
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        # ouput projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config
        self.device = None

    def to(self, device: str):
        self.device = device
        return super().to(device)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequnce of length {T}, block size (max sequence length) is {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (B, T)
        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def speak(self, prefix_str: str, *, num_return_sequences: int = 5, max_length: int = 30, top_k: int = 50):
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(prefix_str)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        x = tokens.to(self.device)  # (B, T)

        while (x.shape[1]) < max_length:
            with torch.no_grad():
                # get logits
                logits, _ = self(x)  # (B, T, vocab_size)
                # we are interested only the in the last token
                logits = logits[:, -1, :]  # (B, vocab_size)
                # probs
                probs = F.softmax(logits, dim=-1)
                # keep only the top-k
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                # sample a token from topk
                ix = torch.multinomial(topk_probs, 1)  # (B, 1) -> index of sample in vocabulary
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                x = torch.cat((x, xcol), dim=1)  # (B, T+1)

        return [enc.decode(pred[:max_length].tolist()) for pred in x]

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model from huggingface"""

        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        print(f"loading pretrained weights from hugging face for {model_type}")
        gpt_config = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        gpt_config["block_size"] = 1024
        gpt_config["vocab_size"] = 50257
        from transformers import GPT2LMHeadModel

        model = GPT(Config(**gpt_config))
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]  # discard this mask/buffer

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        # basically the openai weights are Conv1D, but we only want to use a vanilla Linear
        # this means maually transposing them when we import them
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # transpose these Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epoch = 5000
    learning_rate = 3e-4

    train_loader = DataLoaderLite(B=4, T=32)

    model = GPT(Config())
    model.to(device)

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        for i in range(n_epoch):
            x, y = train_loader.next_batch()
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            print(f"step {i}: loss: {loss.item()}")
    except KeyboardInterrupt:
        pass
    finally:
        model.eval()

        prefix_str = "Hello, I am a language model,"
        preds = model.speak(prefix_str)

        # print the generated text
        for decoded in preds:
            print(f"> {decoded}")
