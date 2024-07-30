import torch
from pathlib import Path
from torch.nn import functional as F

from gpt2 import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

# model = GPT.from_pretrained("gpt2")
res_dir = Path(f"{__file__}/../../exps/").resolve()
chk = res_dir / "exp_lambda" / "last_raw_model_params.pt"
model = GPT.from_checkpoint(checkpoint=chk)
model.eval()
model.to(device)

model_hf = GPT.from_pretrained("gpt2")
model_hf.eval()
model_hf.to(device)

while True:
    prefix_str = input("Say something...: ")
    preds = model.speak(prefix_str, ddp_rank=0, num_return_sequences=1, max_length=50)
    preds_hf = model_hf.speak(prefix_str, ddp_rank=0, num_return_sequences=1, max_length=50)

    # print the generated text
    print("Mine: ")
    for decoded in preds:
        print(f"> {decoded}")
    print()
    print("Hugging face: ")
    for decoded in preds_hf:
        print(f"> {decoded}")
    print()
    print()
