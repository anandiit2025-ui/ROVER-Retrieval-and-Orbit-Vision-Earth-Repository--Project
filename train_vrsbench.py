import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from images import MEAN, STD
from model import build_model


ROOT = Path(__file__).parent
EARTHDIAL = ROOT.parent.parent / "EarthDial"


def load_rows(path, phase):
    rows = json.loads(path.read_text())
    tag = "[caption]" if phase == "caption" else "[vqa]"
    result = []
    for row in rows:
        turns = row.get("conversations", [])
        if len(turns) < 2 or turns[0].get("from") != "human" or turns[1].get("from") != "gpt":
            continue
        question = turns[0].get("value", "")
        answer = turns[1].get("value", "").strip()
        if tag not in question.lower() or not answer:
            continue
        question = re.sub(r"<image>|\[caption\]|\[vqa\]", "", question, flags=re.I).strip()
        result.append((row["image"], question, answer))
    if not result:
        raise ValueError(f"No {phase} examples found in {path}")
    names = sorted({row[0] for row in result})
    random.Random(42).shuffle(names)
    validation = set(names[:max(1, len(names) // 20)])
    train = [row for row in result if row[0] not in validation]
    val = [row for row in result if row[0] in validation]
    random.Random(42).shuffle(train)
    return train, val


def image_tensor(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
        data = torch.from_numpy(np.asarray(image, dtype="float32").copy())
    data = data.permute(2, 0, 1) / 255
    return ((data - MEAN) / STD).unsqueeze(0)


def loss_for(model, image, question, answer):
    prompt = model.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer the question using the satellite image."},
            {"role": "user", "content": f"Image: <image>\nQuestion: {question}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    first, last = prompt.split("<image>", 1)
    embedding = model.qwen.get_input_embeddings()
    device = embedding.weight.device
    dtype = embedding.weight.dtype

    def text_tokens(text):
        ids = model.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        return embedding(ids.to(device)).squeeze(0)

    prefix = torch.cat([
        text_tokens(first),
        model.visual_tokens(image).to(device=device, dtype=dtype),
        text_tokens(last),
    ])
    ids = model.tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    ids = torch.cat([ids, ids.new_tensor([model.tokenizer.eos_token_id])])
    inputs = torch.cat([prefix, embedding(ids)]).unsqueeze(0)
    labels = torch.cat([ids.new_full((len(prefix),), -100), ids]).unsqueeze(0)
    attention = torch.ones(inputs.shape[:2], device=device, dtype=torch.long)
    return model.qwen(inputs_embeds=inputs, attention_mask=attention, labels=labels).loss


def checkpoint(model, optimizer, scaler, output, step, phase):
    path = output / "steps" / f"step_{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    model.qwen.save_pretrained(path / "qwen", safe_serialization=True)
    torch.save(model.projector.state_dict(), path / "projector.pt")
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, path / "training.pt")
    (path / "run.json").write_text(json.dumps({"qwen": "Qwen/Qwen2.5-1.5B-Instruct", "phase": phase, "step": step, "grid": 8}))
    (output / "state.json").write_text(json.dumps({"phase": phase, "step": step, "checkpoint": str(path)}))
    print(f"Saved checkpoint: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["caption", "vqa"], required=True)
    parser.add_argument("--data", type=Path, default=Path("/content/vrsbench"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--steps", type=int, required=True, help="Total optimizer steps, including any completed steps")
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    if args.steps < 1 or args.accumulation < 1 or args.save_every < 1:
        parser.error("steps, accumulation and save-every must be positive")
    image_dir = args.data / "Images_train"
    train, val = load_rows(args.data / "VRSBench_train.json", args.phase)
    missing = [name for name, _, _ in train + val if not (image_dir / name).is_file()]
    if missing:
        parser.error(f"Missing {len(missing)} images; first missing: {missing[0]}")
    print(f"{args.phase}: {len(train)} train, {len(val)} validation; one epoch is {math.ceil(len(train) / args.accumulation)} steps", flush=True)

    state_file = args.output / "state.json"
    start = 0
    adapter = args.adapter
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state["phase"] != args.phase:
            parser.error("Output belongs to another phase")
        start = state["step"]
        adapter = Path(state["checkpoint"])
    elif args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output is not empty: {args.output}")
    if adapter and (adapter / "state.json").exists():
        adapter = Path(json.loads((adapter / "state.json").read_text())["checkpoint"])
    if args.steps <= start:
        print(f"Already at step {start}; increase --steps to continue", flush=True)
        return
    if not torch.cuda.is_available():
        parser.error("Select a GPU runtime in Colab")

    model = build_model(
        ROOT / "vision_checkpoint", EARTHDIAL, "Qwen/Qwen2.5-1.5B-Instruct",
        train=True, adapter=adapter, grid=8,
    )
    model.train()
    model.vision.eval()
    if args.phase == "caption":
        for param in model.qwen.parameters():
            param.requires_grad_(False)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4 if args.phase == "caption" else 1e-4)
    scaler = torch.amp.GradScaler("cuda")
    if start:
        saved = torch.load(adapter / "training.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = args.output / "metrics.csv"
    if not metrics.exists():
        with metrics.open("w", newline="") as file:
            csv.writer(file).writerow(["step", "train_loss", "validation_loss", "gpu_reserved_gb"])

    for step in range(start + 1, args.steps + 1):
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(args.accumulation):
            name, question, answer = train[((step - 1) * args.accumulation + i) % len(train)]
            image = image_tensor(image_dir / name)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_for(model, image, question, answer)
            scaler.scale(loss / args.accumulation).backward()
            total += loss.item() / args.accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()

        check = step % args.save_every == 0 or step == args.steps
        val_loss = None
        if check:
            model.eval()
            with torch.no_grad():
                losses = []
                for name, question, answer in val[:8]:
                    image = image_tensor(image_dir / name)
                    with torch.autocast("cuda", dtype=torch.float16):
                        losses.append(loss_for(model, image, question, answer).item())
            val_loss = sum(losses) / len(losses)
            model.train()
            model.vision.eval()
        gpu = torch.cuda.max_memory_reserved() / 1024 ** 3
        with metrics.open("a", newline="") as file:
            csv.writer(file).writerow([step, total, val_loss, gpu])
        if step == 1 or step % 10 == 0 or check:
            print(f"Step {step}/{args.steps}: loss {total:.4f}; val {val_loss}; GPU {gpu:.1f} GB", flush=True)
        if check:
            checkpoint(model, optimizer, scaler, args.output, step, args.phase)


if __name__ == "__main__":
    main()
