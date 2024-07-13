import torch
import tiktoken
from pathlib import Path


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes

        self.current_pos = self.B * self.T * self.process_rank

        with open(Path(f"{__file__}/../../data/input.txt").resolve(), "r") as f:
            _text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(_text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens):_} tokens")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos : self.current_pos + (B * T + 1)]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_pos += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_pos = self.B * self.T * self.process_rank
        return x, y
