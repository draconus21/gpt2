import os
from pathlib import Path
import multiprocessing as mp

import numpy as np

import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = str(Path(f"{__file__}/../../data/edu_fineweb10B").resolve())
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard, total on 100 shards

# create the cache
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np <= 2**16
    ).all(), f"token dictionary too large: min: {tokens_np.min()}, max: {tokens_np.max()}"

    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


nprocs = max(1, os.cpu_count() // 2)

with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # if tehre enough space in the current shard for the new tokes?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(local_dir, f"edufileweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to the next one
            remainder = shard_size - token_count
            all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            progress_bar.update(remainder)

            shard_index += 1
            progress_bar = None  # populate the next shard w/ the left overs of the current doc
            all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(local_dir, f"edufileweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
