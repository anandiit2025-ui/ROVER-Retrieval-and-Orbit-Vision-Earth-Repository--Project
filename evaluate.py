import argparse
import json
from pathlib import Path

from data import ChangePairs, FPCDPairs, SETS
from images import prepare_pair
from model import build_model


ROOT = Path(__file__).parent
EARTHDIAL = ROOT.parent.parent / "EarthDial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--set", choices=[*SETS, "fpcd"], default="levir")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation.json")
    args = parser.parse_args()

    settings = json.loads((args.adapter / "run.json").read_text())
    if args.set == "fpcd":
        data = FPCDPairs(ROOT / "data" / "fpcd", validation=True)
    else:
        path = ROOT / "data" / "training_set" / SETS[args.set][0]
        data = ChangePairs(path, validation=True)
    model = build_model(
        ROOT / "vision_checkpoint" if (ROOT / "vision_checkpoint").exists()
        else EARTHDIAL / "checkpoints" / "EarthDial_4B_RGB", EARTHDIAL,
        settings["qwen"], adapter=args.adapter,
    ).eval()
    result = []
    for i in range(min(args.count, len(data))):
        row = data[i]
        before, after, _ = prepare_pair(
            row["before"], row["after"], max_tiles=settings["max_tiles"],
        )
        prediction = model.answer(before, after, row["question"])
        result.append({
            "key": row["key"],
            "question": row["question"],
            "expected": row["answer"],
            "predicted": prediction,
        })
        print(f"{i + 1}/{min(args.count, len(data))}: {prediction}", flush=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
