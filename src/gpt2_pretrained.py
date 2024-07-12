import torch
from torch.nn import functional as F

from gpt2 import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)


prefix_str = "Hello, I am a language model,"
preds = model.speak(prefix_str)

# print the generated text
for decoded in preds:
    print(f"> {decoded}")
