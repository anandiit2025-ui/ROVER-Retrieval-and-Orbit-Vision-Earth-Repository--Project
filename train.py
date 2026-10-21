import argparse
import csv
import json
import random
from pathlib import Path

import torch

from data import ChangePairs, FPCDPairs, SETS, download, download_fpcd
from images import prepare_pair
from model import build_model
from plot import plot_metrics


ROOT = Path(__file__).parent
EARTHDIAL = ROOT.parent.parent / "EarthDial"


def dataset_path(name):
    return ROOT / "data" / "training_set" / SETS[name][0]


def save_checkpoint(model, folder, settings):
    folder.mkdir(parents=True, exist_ok=True)
    model.qwen.save_pretrained(folder / "qwen", safe_serialization=True)
    torch.save(model.projector.state_dict(), folder / "projector.pt")
    (folder / "run.json").write_text(json.dumps(settings, indent=2))
    print(f"Saved checkpoint: {folder}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=[*SETS, "fpcd"], default="dubai")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-tiles", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--eval-count", type=int, default=3)
    parser.add_argument("--min-free-gb", type=float, default=16)
    parser.add_argument("--qwen", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--vision-checkpoint", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "first_run")
    args = parser.parse_args()
    if args.steps < 1 or args.save_every < 1 or args.eval_count < 1:
        parser.error("Steps, save-every, and eval-count must be positive")

    if args.set == "fpcd":
        path = download_fpcd(ROOT / "data" / "fpcd") if args.download else ROOT / "data" / "fpcd"
        if not (path / "multi_class_masks" / "train").exists():
            parser.error(f"Dataset is missing: {path}. Run with --download.")
        data = FPCDPairs(path, limit=args.limit)
        validation = FPCDPairs(path, limit=args.limit, validation=True)
    else:
        path = download(args.set, ROOT / "data") if args.download else dataset_path(args.set)
        if not path.exists():
            parser.error(f"Dataset is missing: {path}. Run with --download.")
        data = ChangePairs(path, limit=args.limit)
        validation = ChangePairs(path, limit=args.limit, validation=True)
    print(f"Training examples: {len(data)}; validation examples: {len(validation)}", flush=True)
    if not len(data):
        parser.error("No training examples are available")
    sample = data[0]
    before, after, boxes = prepare_pair(
        sample["before"], sample["after"], max_tiles=args.max_tiles,
    )
    print(f"Image pair: {sample['before'].size}; tiles per image: {len(boxes)}", flush=True)
    print(f"Sample question: {sample['question']}", flush=True)
    if args.data_only:
        return

    if args.resume_from:
        if not (args.resume_from / "projector.pt").exists():
            parser.error(f"Adapter is missing: {args.resume_from}")
        if args.output.resolve() == args.resume_from.resolve():
            parser.error("Output must differ from --resume-from")
        previous = json.loads((args.resume_from / "run.json").read_text())
        if previous["qwen"] != args.qwen:
            parser.error("Qwen model must match the previous run")
    if (args.output / "metrics.csv").exists():
        parser.error(f"Output already has a run: {args.output}. Use a new output folder.")

    if not torch.cuda.is_available():
        parser.error("CUDA is required for this training run")
    free, _ = torch.cuda.mem_get_info()
    if free / 1024 ** 3 < args.min_free_gb:
        parser.error(
            f"Only {free / 1024 ** 3:.1f} GB GPU memory is free; "
            f"need {args.min_free_gb:g} GB. Other jobs were left untouched."
        )

    vision = args.vision_checkpoint or ROOT / "vision_checkpoint"
    if not vision.exists():
        vision = EARTHDIAL / "checkpoints" / "EarthDial_4B_RGB"
    model = build_model(
        vision, EARTHDIAL,
        args.qwen, train=True, adapter=args.resume_from,
    )
    model.train()
    model.vision.eval()
    params = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    scaler = torch.cuda.amp.GradScaler()
    rng = random.Random(42)
    optimizer.zero_grad(set_to_none=True)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = args.output / "metrics.csv"
    with metrics.open("w", newline="") as file:
        csv.writer(file).writerow(["step", "train_loss", "validation_loss", "gpu_reserved_gb"])

    for step in range(1, args.steps + 1):
        torch.cuda.reset_peak_memory_stats()
        loss_total = 0
        for _ in range(args.accumulation):
            sample = data[rng.randrange(len(data))]
            before, after, _ = prepare_pair(
                sample["before"], sample["after"], max_tiles=args.max_tiles,
            )
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(before, after, sample["question"], sample["answer"])
            scaler.scale(loss / args.accumulation).backward()
            loss_total += float(loss.detach()) / args.accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
        save = step % args.save_every == 0 or step == args.steps
        validation_loss = None
        if save and len(validation):
            model.eval()
            losses = []
            with torch.no_grad():
                for i in range(min(args.eval_count, len(validation))):
                    sample = validation[i]
                    before, after, _ = prepare_pair(
                        sample["before"], sample["after"], max_tiles=args.max_tiles,
                    )
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = model(before, after, sample["question"], sample["answer"])
                    losses.append(float(loss))
            validation_loss = sum(losses) / len(losses)
            model.train()
            model.vision.eval()
        with metrics.open("a", newline="") as file:
            csv.writer(file).writerow([step, loss_total, validation_loss, reserved])
        if step == 1 or step % 10 == 0 or save:
            print(f"Step {step}/{args.steps}: loss {loss_total:.4f}; GPU {reserved:.1f} GB", flush=True)
        if save:
            result = {
                "qwen": args.qwen,
                "dataset": args.set,
                "step": step,
                "steps": args.steps,
                "max_tiles": args.max_tiles,
                "validation_loss": validation_loss,
                "resumed_from": str(args.resume_from) if args.resume_from else None,
            }
            save_checkpoint(model, args.output / "steps" / f"step_{step:06d}", result)
            plot_metrics(metrics)

    save_checkpoint(model, args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
