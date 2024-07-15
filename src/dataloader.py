import os
import torch
import tiktoken
import numpy as np
from pathlib import Path


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {"train", "val"}

        # get shard file names
        data_root = Path(f"{__file__}/../../data/edufileweb10B")
        self.shards = sorted([os.path.join(data_root, s) for s in os.listdir(data_root) if split in s])

        assert len(self.shards) > 0, f"no shards found for split {split}"

        # state, init at shard 0
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos : self.current_pos + (B * T + 1)]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_pos += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = self.B * self.T * self.process_rank
        return x, y
