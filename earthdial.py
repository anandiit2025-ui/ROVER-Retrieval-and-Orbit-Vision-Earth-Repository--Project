import base64
import json
import os
from io import BytesIO

import requests
import numpy as np
import rasterio
from PIL import Image

from utils import ROOT, RoverError


def earthdial_configured():
    return bool(os.getenv("EARTHDIAL_WORKER_URL", "").strip())


def encode_image(relative_path, limits=None):
    path = ROOT / relative_path
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise RoverError(f"EarthDial accepts TIFF inputs only: {relative_path}")
    try:
        with rasterio.open(path) as source:
            data = source.read(masked=True).astype("float32").filled(np.nan)
        if data.shape[0] not in {1, 3}:
            raise ValueError(f"Expected one or three TIFF bands, got {data.shape[0]}")
        valid = np.all(np.isfinite(data), axis=0)
        if not np.any(valid):
            raise ValueError("TIFF has no valid pixels")
        if path.name.startswith("s2_") and path.name.endswith("_rgb.tif"):
            pixels = np.power(np.clip(data / 3000, 0, 1), 0.8)
            pixels[:, ~valid] = 0.85
        elif limits is not None:
            minimum, maximum = limits
        elif path.name.startswith(("ndwi_", "ndvi_", "nbr_")):
            minimum, maximum = -1, 1
        else:
            minimum, maximum = np.percentile(data[:, valid], [2, 98])
        if not (path.name.startswith("s2_") and path.name.endswith("_rgb.tif")):
            if maximum <= minimum:
                maximum = minimum + 1
            pixels = np.clip((data - minimum) / (maximum - minimum), 0, 1)
            pixels = np.nan_to_num(pixels, nan=0.5)
            pixels[:, ~valid] = 0.5
        if data.shape[0] == 1:
            pixels = np.repeat(pixels, 3, axis=0)
        pixels = np.moveaxis(np.rint(pixels * 255).astype("uint8"), 0, -1)
        buffer = BytesIO()
        Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    except (OSError, ValueError) as error:
        raise RoverError(f"Could not prepare EarthDial TIFF {relative_path}: {error}") from error
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_prompt(inputs, question, intent, evidence=None):
    evidence = evidence or {}
    return "\n".join([
        "Image 1 is before and Image 2 is after.",
        f"User question: {question}",
        "Independent raster measurements (estimates, not visual facts): "
        f"{json.dumps(evidence.get('measurements', {}))}",
        f"Image quality: {json.dumps(evidence.get('quality', {}))}",
        "Use the images and measurements to answer the question. Distinguish measured "
        "change from what is visibly clear. Mention uncertainty from clouds or missing "
        "pixels. Do not invent landmarks, causes, or exact areas beyond the measurements.",
    ])


def infer_earthdial(inputs, question, intent, evidence=None):
    worker_url = os.getenv("EARTHDIAL_WORKER_URL", "").strip().rstrip("/")
    if not worker_url:
        raise RoverError("EARTHDIAL_WORKER_URL is not configured")

    try:
        max_tiles = int(os.getenv("EARTHDIAL_MAX_INPUT_TILES", "12"))
    except ValueError as error:
        raise RoverError("EARTHDIAL_MAX_INPUT_TILES must be an integer") from error
    if max_tiles < 1:
        raise RoverError("EARTHDIAL_MAX_INPUT_TILES must be positive")

    model = os.getenv("EARTHDIAL_MODEL", "EarthDial_4B_RGB")
    optical = [item for item in inputs if item["path"].endswith("_rgb.tif")]
    if len(optical) != 2:
        raise RoverError("EarthDial RGB model needs two Sentinel-2 RGB images")
    inputs = optical
    interpretations = []
    batch_size = 2
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        payload = {
            "model": model,
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Earth-observation assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": build_prompt(batch, question, intent, evidence),
                    "image": [
                        encode_image(item["path"]) for item in batch
                    ],
                },
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 150,
            "max_input_tiles": max_tiles,
            "repetition_penalty": 1.0,
        }

        try:
            response = requests.post(
                f"{worker_url}/worker_generate_stream",
                json=payload,
                headers={"User-Agent": "ROVER EarthDial Client"},
                stream=True,
                timeout=(10, 300),
            )
            response.raise_for_status()
            generated_text = None
            for chunk in response.iter_lines(delimiter=b"\0"):
                if not chunk:
                    continue
                message = json.loads(chunk.decode("utf-8"))
                if message.get("error_code") != 0:
                    raise RoverError(message.get("text", "EarthDial inference failed"))
                generated_text = message.get("text")
        except (requests.RequestException, UnicodeError, json.JSONDecodeError) as error:
            raise RoverError(f"EarthDial request failed: {error}") from error

        if not generated_text:
            raise RoverError("EarthDial returned no text")
        names = ", ".join(item["label"] for item in batch)
        interpretations.append(f"Rendered images {names}: {generated_text.strip()}")
    return {
        "model": model,
        "basis": "rendered before/after images and raster measurements",
        "interpretation": "\n\n".join(interpretations),
        "images": inputs,
    }
