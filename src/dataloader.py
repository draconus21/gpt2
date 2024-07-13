import torch
import tiktoken
from pathlib import Path


class DataLoaderLite:
    def __init__(self, B, T):
        self.B, self.T = B, T
        self.current_pos = 0

        with open(Path(f"{__file__}/../../data/input.txt").resolve(), "r") as f:
            _text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(_text)

        # trim to multiple of B*T
        n = len(tokens)
        excess = n % (self.B * self.T)
        tokens = tokens[: -(excess - 1)]  # Keep %(B*T)  + 1
        print(f"discarding {excess} out of {n:_} tokens")
        self.tokens = torch.tensor(tokens)

        print(f"loaded {len(self.tokens):_} tokens")
        print(f"1 epoch = {len(self.tokens[:-1])/(self.B * self.T)} batches")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos : self.current_pos + (B * T + 1)]
        try:
            x = buf[:-1].view(B, T)
            y = buf[1:].view(B, T)
        except Exception as e:
            print(f"current: {self.current_pos}, B: {B}, T:{T}, len buf: {buf.shape}, len tok: {self.tokens.shape}")
            raise e

        self.current_pos = (self.current_pos + B * T) % len(self.tokens - 1)
        if len(self.tokens) - self.current_pos == 1:  # last one
            self.current_pos = 0
        return x, y
