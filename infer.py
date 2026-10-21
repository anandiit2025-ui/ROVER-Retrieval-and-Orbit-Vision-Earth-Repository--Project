import argparse
import json
from pathlib import Path

from images import open_image, prepare_pair
from model import build_model


ROOT = Path(__file__).parent
ROVER = ROOT.parent
EARTHDIAL = ROVER.parent / "EarthDial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--rover-output", type=Path, default=ROVER / "build" / "output.json")
    parser.add_argument("--output", type=Path, default=ROVER / "build" / "custom_earthdial_output.json")
    args = parser.parse_args()

    run = json.loads(args.rover_output.read_text())
    evidence = run["evidence"]
    rasters = evidence["rasters"]
    if "s2_before_rgb" not in rasters or "s2_after_rgb" not in rasters:
        parser.error("This run has no Sentinel-2 before/after RGB rasters")
    before = open_image(ROVER / rasters["s2_before_rgb"])
    after = open_image(ROVER / rasters["s2_after_rgb"])
    settings = json.loads((args.adapter / "run.json").read_text())
    first, second, boxes = prepare_pair(
        before, after, max_tiles=settings["max_tiles"],
    )
    model = build_model(
        ROOT / "vision_checkpoint" if (ROOT / "vision_checkpoint").exists()
        else EARTHDIAL / "checkpoints" / "EarthDial_4B_RGB", EARTHDIAL,
        settings["qwen"], adapter=args.adapter,
    ).eval()
    answer = model.answer(
        first, second, run["question"], evidence["measurements"], evidence["quality"],
    )
    visuals = evidence.get("visuals", {})
    references = {
        name: str(ROVER / visuals[name])
        for name in ("before_image", "after_image", "result_image")
        if name in visuals
    }
    result = {
        "question": run["question"],
        "answer": answer,
        "reference_images": references,
        "measurements": evidence["measurements"],
        "quality": evidence["quality"],
        "tiles_per_image": len(boxes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(answer)
    print("Reference images:")
    for name, path in references.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
