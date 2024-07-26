import math
import torch
import inspect
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
        # jank for init std scaling for residual pathways
        self.c_proj.NANOGPT_USES_RESIDUAL = 1

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
        # attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        ## mask the future
        # attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        # jank for init std scaling for residual pathways
        self.c_proj.NANOGPT_USES_RESIDUAL = 1

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

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # openai's init
        self.apply(self._init_weights)

    def to(self, device: str):
        self.device = device
        return super().to(device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_USES_RESIDUAL"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

    def speak(
        self, prefix_str: str, ddp_rank: int, *, num_return_sequences: int = 5, max_length: int = 30, top_k: int = 50
    ):
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(prefix_str)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        x = tokens.to(self.device)  # (B, T)

        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(42 + ddp_rank)

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
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1) -> index of sample in vocabulary
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                x = torch.cat((x, xcol), dim=1)  # (B, T+1)

        return [enc.decode(pred[:max_length].tolist()) for pred in x]

    def speak_to_file(self, fname: str, prefix_str: str, ddp_rank: int, **kwargs):
        preds = raw_model.speak(prefix_str, ddp_rank, **kwargs)
        with open(fname, "w") as f:
            for pred in preds:
                f.writelines(f"> {pred}\n")
        return preds

    def configure_optimizers(self, weight_decay, learning_rate, device, **kwargs):
        # start with all candidate params that require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # create optim groups. Any param that is 2+D will be weight decayed, otherwise no.
        # i.e. all weight tensorsin matmuls + embeddings decay, all biases and layer norms don't
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed param tensors    : {len(decay_params):3d}, with {num_decay_params:_} params")
        print(f"num non-decayed param tensors: {len(nodecay_params):3d}, with {num_nodecay_params:_} params")

        # create the adam optimizer and use fused if available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        fused = fused_available and "cuda" in device
        print(f"Using fused AdamW: {fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=fused, **kwargs)
        return optimizer

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


def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LRScheduler:
    def __init__(self, max_lr, min_lr, warmup_steps, max_steps):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step):
        # 1. linear warmup
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        # 2. if step > lr_decal_steps, return min lr
        if step > self.max_steps:
            return self.min_lr

        # 3. in between the two, use cosine decay down to min_lr
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and decays to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


if __name__ == "__main__":
    import os
    import time
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    from torch.distributed import init_process_group, destroy_process_group

    # set up DDP (Distributed Data Parallel)
    # torchrun sets the env vars RANK, LOCAL_RANK, WORLD_SIZE
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        assert torch.cuda.is_available(), "We need CUDA for DDP"
        init_process_group("nccl")

        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will additionally do logging, checkpointing, etc.
    else:
        # vanilla, non-DDP
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # autodetect cuda
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        print(f"using device: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # grad accumulation
    total_batch_size = 2**19  # ~0.5M tokens = ~0.5M / 50304
    B, T = 2, 1024  # micro_batch, seq length
    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), f"total_batch_size [{total_batch_size}] not divisible by B*T*ddp_world_size [{B}*{T}*{ddp_world_size}={B*T*ddp_world_size}]"

    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    prefix_str = "Hello, I am a language model,"
    max_length = 30

    res_dir = str(Path(f"{__file__}/../../exps/").resolve())
    n = len(os.listdir(res_dir)) if os.path.exists(res_dir) else 0
    res_dir = Path(res_dir) / f"exp_{n}"
    os.makedirs(res_dir)

    best_model_params_path = (Path(res_dir) / "best_params.pt").resolve()
    best_rmodel_params_path = (Path(res_dir) / "best_raw_model_params.pt").resolve()
    best_loss = float("inf")

    def _save_best(last=False):
        if last:
            print("saving last")
        print(f"best params [loss: {best_loss:.3f}] saved to: {best_model_params_path}")

        torch.save(
            model.state_dict(),
            Path(str(best_model_params_path).replace("best", "last")) if last else best_model_params_path,
        )
        torch.save(
            raw_model.state_dict(),
            Path(str(best_rmodel_params_path).replace("best", "last")) if last else best_rmodel_params_path,
        )

    valid_freq = 100
    n_val_step = 20
    n_epoch = 19073  # 10e9/2**19
    warmup_steps = 715
    max_lr = 6e-4
    weight_decay = 0.1
    learning_rate = LRScheduler(max_lr=max_lr, min_lr=0.1 * max_lr, warmup_steps=warmup_steps, max_steps=n_epoch)
    betas = (0.9, 0.95)
    eps = 1e-8

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
    if master_process:
        print("n_shards:", {l.split: len(l.shards) for l in [train_loader, val_loader]})

    torch.set_float32_matmul_precision("high")  # no change

    model = GPT(Config(vocab_size=50304))
    model.to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # gpu is too old for it :(
    model = torch.compile(model)

    try:
        print(f"# trainable params: {count_parameters(model):_}")
        optimizer = raw_model.configure_optimizers(
            weight_decay=weight_decay, learning_rate=3e-4, device=device, betas=betas, eps=eps
        )
        for i in range(n_epoch):
            t0 = time.time()

            if i % valid_freq == 1:
                # do validation
                model.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    for _ in range(n_val_step):
                        x, y = val_loader.next_batch()
                        x, y = x.to(device), y.to(device)
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):  # made it worse
                            logits, loss = model(x, y)
                        loss = loss / n_val_step
                        val_loss_accum += loss.detach()
                    if ddp:
                        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    print(f"validation loss: {val_loss_accum.item():.4f}")
                    if val_loss_accum.item() < best_loss:
                        best_loss = val_loss_accum.item()
                        _save_best()
                    # generate data
                    fname = str(res_dir / f"gpt2_{i-1}_val_{val_loss_accum.item():.4f}.txt")
                    raw_model.speak_to_file(fname, prefix_str, ddp_rank, max_length=max_length)

            # training loop
            model.train()
            optimizer.zero_grad()

            loss_accum = 0.0
            for micro_batch in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x = x.to(device)
                y = y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):  # made it worse
                    logits, loss = model(x, y)
                loss_accum += loss.detach()

                loss = loss / total_batch_size
                if ddp:
                    model.require_backward_grad_sync = micro_batch == grad_accum_steps - 1
                loss.backward()

            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = learning_rate.get_lr(i)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.step()
            torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000  # ms
            tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1 - t0)
            if master_process:
                print(
                    f"step {i:3d}: loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.1f}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        if ddp:
            destroy_process_group()
        if master_process:
            model.eval()

            _save_best(last=True)
            fname = str(res_dir / f"gpt2_{i-1}.txt")
            preds = raw_model.speak_to_file(fname, prefix_str, ddp_rank, max_length=max_length)

            # print the generated text
            for decoded in preds:
                print(f"> {decoded}")
