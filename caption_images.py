"""Caption two ROVER RGB images independently with the trained projector."""

import argparse
import json
from pathlib import Path

import torch

from images import open_image, prepare_pair
from model import build_model


ROOT = Path(__file__).parent
EARTHDIAL = ROOT.parent.parent / "EarthDial"
CAPTION_QUESTION = "Please provide a detailed description of the image."


def caption(model, image):
    tokenizer = model.tokenizer
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer the question using the satellite image."},
            {"role": "user", "content": f"Image: <image>\nQuestion: {CAPTION_QUESTION}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    first, last = prompt.split("<image>", 1)
    embedding = model.qwen.get_input_embeddings()
    device = embedding.weight.device
    dtype = embedding.weight.dtype

    def text_tokens(text):
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        return embedding(ids.to(device)).squeeze(0)

    prefix = torch.cat([
        text_tokens(first),
        model.visual_tokens(image).to(device=device, dtype=dtype),
        text_tokens(last),
    ]).unsqueeze(0)
    attention = torch.ones(prefix.shape[:2], device=device, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.float16):
        generated = model.qwen.generate(
            inputs_embeds=prefix,
            attention_mask=attention,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    if not result:
        raise RuntimeError("Qwen generated an empty caption")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-datetime", default="")
    parser.add_argument("--after-datetime", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    settings = json.loads((args.adapter / "run.json").read_text(encoding="utf-8"))
    if settings.get("phase") != "caption":
        parser.error("Adapter must be a caption-phase checkpoint")
    before, after, _ = prepare_pair(
        open_image(args.before), open_image(args.after), max_tiles=1,
    )
    model = build_model(
        ROOT / "vision_checkpoint", EARTHDIAL, settings["qwen"],
        train=False, adapter=args.adapter, grid=settings["grid"],
    ).eval()
    with torch.inference_mode():
        images = [
            {
                "role": "before", "source": str(args.before),
                "datetime": args.before_datetime, "caption": caption(model, before),
            },
            {
                "role": "after", "source": str(args.after),
                "datetime": args.after_datetime, "caption": caption(model, after),
            },
        ]
    result = {
        "schema_version": 1,
        "checkpoint": str(args.adapter),
        "caption_model": settings["qwen"],
        "images": images,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"Saved captions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
