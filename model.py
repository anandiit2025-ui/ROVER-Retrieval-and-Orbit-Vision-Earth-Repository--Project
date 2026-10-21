import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn


SYSTEM = (
    "Compare the two satellite images. Describe only changes supported by the images. "
    "Use supplied raster measurements as estimates. If image interpretation and "
    "measurements disagree, state the disagreement. Do not invent causes."
)


def load_vision(checkpoint, repo, device):
    sys.path.insert(0, str(Path(repo) / "src"))
    from earthdial.model.internvl_chat.configuration_intern_vit import InternVisionConfig
    from earthdial.model.internvl_chat.modeling_intern_vit import InternVisionModel

    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    vision_config = InternVisionConfig(**config["vision_config"])
    vision_config.use_flash_attn = False
    vision = InternVisionModel(vision_config)
    weights = {}
    portable = checkpoint / "vision.safetensors"
    if portable.exists():
        with safe_open(portable, framework="pt", device="cpu") as source:
            for key in source.keys():
                weights[key] = source.get_tensor(key)
    else:
        for shard in sorted(checkpoint.glob("model-*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as source:
                for key in source.keys():
                    if key.startswith("vision_model."):
                        weights[key.removeprefix("vision_model.")] = source.get_tensor(key)
    if not weights:
        raise ValueError("The EarthDial checkpoint has no InternViT weights")
    vision.load_state_dict(weights, strict=True)
    del weights
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    vision.to(device=device, dtype=dtype).eval()
    vision.requires_grad_(False)
    return vision


class EarthDialQwen(nn.Module):
    def __init__(self, vision, qwen, tokenizer, grid=4):
        super().__init__()
        self.vision = vision
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.grid = grid
        hidden = qwen.config.hidden_size
        self.projector = nn.Sequential(
            nn.LayerNorm(vision.config.hidden_size),
            nn.Linear(vision.config.hidden_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.projector.to(next(qwen.parameters()).device)

    def visual_tokens(self, tiles):
        device = next(self.vision.parameters()).device
        dtype = next(self.vision.parameters()).dtype
        features = []
        for tile in tiles:
            with torch.no_grad():
                output = self.vision(
                    pixel_values=tile.unsqueeze(0).to(device=device, dtype=dtype),
                    return_dict=True,
                ).last_hidden_state[:, 1:, :]
                side = int(output.shape[1] ** 0.5)
                if side * side != output.shape[1]:
                    raise ValueError("InternViT returned a non-square patch grid")
                output = output.transpose(1, 2).reshape(1, -1, side, side)
                output = F.adaptive_avg_pool2d(output, (self.grid, self.grid))
                features.append(output.flatten(2).transpose(1, 2).squeeze(0))
        data = torch.cat(features).float().to(next(self.projector.parameters()).device)
        return self.projector(data)

    def prompt_embeds(self, before, after, question, measurements=None, quality=None):
        content = (
            f"Question: {question}\n"
            f"Raster measurements: {json.dumps(measurements or {})}\n"
            f"Scene quality: {json.dumps(quality or {})}\n"
            "Before image: <before_image>\nAfter image: <after_image>\n"
            "Answer briefly and say when the images are inconclusive."
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        first, rest = prompt.split("<before_image>", 1)
        middle, last = rest.split("<after_image>", 1)
        embedding = self.qwen.get_input_embeddings()
        dtype = embedding.weight.dtype
        device = embedding.weight.device

        def text_tokens(text):
            ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
            return embedding(ids.to(device)).squeeze(0)

        parts = [
            text_tokens(first),
            self.visual_tokens(before).to(device=device, dtype=dtype),
            text_tokens(middle),
            self.visual_tokens(after).to(device=device, dtype=dtype),
            text_tokens(last),
        ]
        return torch.cat(parts, dim=0)

    def forward(self, before, after, question, answer, measurements=None, quality=None):
        prefix = self.prompt_embeds(before, after, question, measurements, quality)
        ids = self.tokenizer(
            answer, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].to(prefix.device)
        ids = torch.cat([ids, ids.new_tensor([self.tokenizer.eos_token_id])])
        answer_embeds = self.qwen.get_input_embeddings()(ids)
        inputs = torch.cat([prefix, answer_embeds], dim=0).unsqueeze(0)
        labels = torch.cat([ids.new_full((len(prefix),), -100), ids]).unsqueeze(0)
        attention = torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.long)
        return self.qwen(inputs_embeds=inputs, attention_mask=attention, labels=labels).loss

    @torch.inference_mode()
    def answer(self, before, after, question, measurements=None, quality=None, max_new_tokens=120):
        prefix = self.prompt_embeds(before, after, question, measurements, quality)
        attention = torch.ones((1, len(prefix)), device=prefix.device, dtype=torch.long)
        output = self.qwen(
            inputs_embeds=prefix.unsqueeze(0), attention_mask=attention,
            use_cache=True,
        )
        generated = []
        for _ in range(max_new_tokens):
            token = output.logits[:, -1].argmax(dim=-1)
            if token.item() == self.tokenizer.eos_token_id:
                break
            generated.append(token.item())
            attention = torch.cat([attention, attention.new_ones((1, 1))], dim=1)
            output = self.qwen(
                input_ids=token.unsqueeze(0), attention_mask=attention,
                past_key_values=output.past_key_values, use_cache=True,
            )
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_model(checkpoint, repo, qwen_name, device="cuda", train=False, adapter=None, grid=4):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if str(device).startswith("cuda"):
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        options = {"quantization_config": quantization, "device_map": {"": 0}}
    else:
        options = {"torch_dtype": torch.float32}

    tokenizer = AutoTokenizer.from_pretrained(qwen_name, use_fast=True)
    qwen = AutoModelForCausalLM.from_pretrained(qwen_name, **options)
    if train:
        if str(device).startswith("cuda"):
            qwen = prepare_model_for_kbit_training(qwen, use_gradient_checkpointing=True)
        qwen.config.use_cache = False
        if adapter:
            qwen = PeftModel.from_pretrained(qwen, str(Path(adapter) / "qwen"), is_trainable=True)
        else:
            qwen = get_peft_model(qwen, LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            ))
    elif adapter:
        qwen = PeftModel.from_pretrained(qwen, str(Path(adapter) / "qwen"))

    vision = load_vision(checkpoint, repo, device)
    model = EarthDialQwen(vision, qwen, tokenizer, grid)
    if adapter:
        state = torch.load(Path(adapter) / "projector.pt", map_location="cpu", weights_only=True)
        model.projector.load_state_dict(state)
    if not str(device).startswith("cuda"):
        model.to(device)
    return model
