import argparse
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    weights = {}
    for shard in sorted(args.source.glob("model-*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as data:
            for key in data.keys():
                if key.startswith("vision_model."):
                    weights[key.removeprefix("vision_model.")] = data.get_tensor(key)
    if not weights:
        parser.error("No InternViT weights found")
    save_file(weights, args.output / "vision.safetensors")
    shutil.copy2(args.source / "config.json", args.output / "config.json")
    print(f"Saved {len(weights)} vision tensors to {args.output}")


if __name__ == "__main__":
    main()
