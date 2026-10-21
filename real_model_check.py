import torch

from data import ChangePairs
from images import prepare_pair
from model import build_model
from train import EARTHDIAL, ROOT, dataset_path


data = ChangePairs(dataset_path("dubai"), limit=10)
row = data[0]
before, after, _ = prepare_pair(row["before"], row["after"], max_tiles=1)
model = build_model(
    ROOT / "vision_checkpoint" if (ROOT / "vision_checkpoint").exists()
    else EARTHDIAL / "checkpoints" / "EarthDial_4B_RGB", EARTHDIAL,
    "Qwen/Qwen2.5-1.5B-Instruct", device="cpu",
).eval()
with torch.inference_mode():
    loss = model(before, after, row["question"], row["answer"])
    answer = model.answer(before, after, row["question"], max_new_tokens=2)
print(f"Real model forward passed; loss={float(loss):.3f}; answer={answer!r}")
