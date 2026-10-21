"""Run the trained RGB caption model in its separate PyTorch environment."""

import json
import os
import subprocess
from pathlib import Path

from utils import ROOT, RoverError


CUSTOM = ROOT / "earthdial_custom"
DEFAULT_CHECKPOINT = CUSTOM / "runs/vrsbench_caption_run1/steps/step_009626"
DEFAULT_MS_CHECKPOINT = CUSTOM / "runs/chatearthnet_ms_caption_20k_run1/steps/step_010000"
DEFAULT_PYTHON = ROOT.parent / "EarthDial/.venv/bin/python"


def generate_captions(rasters, scene=None, multispectral=None, unavailable_reason=None):
    scene = scene or {}
    before = ROOT / rasters["s2_before_rgb"]
    after = ROOT / rasters["s2_after_rgb"]
    for image in (before, after):
        if not image.is_file():
            raise RoverError(f"Caption input is missing: {image}")

    python = Path(os.getenv("CUSTOM_EARTHDIAL_PYTHON", DEFAULT_PYTHON))
    output = ROOT / "build/captions.json"
    if not python.is_file():
        raise RoverError(f"Caption Python environment is missing: {python}")

    fallback_reason = unavailable_reason
    if multispectral:
        checkpoint = Path(os.getenv("CUSTOM_EARTHDIAL_MS_CHECKPOINT", DEFAULT_MS_CHECKPOINT))
        inputs = multispectral["images"]
        if (checkpoint / "run.json").is_file():
            command = [
                str(python), str(CUSTOM / "caption_multispectral.py"),
                "--checkpoint", str(checkpoint),
                "--before", str(ROOT / inputs["before"]["path"]),
                "--after", str(ROOT / inputs["after"]["path"]),
                "--before-datetime", str(scene.get("before", {}).get("datetime", "")),
                "--after-datetime", str(scene.get("after", {}).get("datetime", "")),
                "--output", str(output),
            ]
            try:
                subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               timeout=600, check=True)
                result = json.loads(output.read_text(encoding="utf-8"))
                if result.get("mode") != "multispectral" or len(result.get("images", [])) != 2:
                    raise ValueError("Multispectral runner returned incomplete captions")
                result["preprocessing"] = multispectral["preprocessing"]
                result["clear_fraction"] = {
                    role: inputs[role]["clear_fraction"] for role in ("before", "after")
                }
                from utils import save_json
                save_json("build/captions.json", result)
                return result
            except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError,
                    ValueError, json.JSONDecodeError) as error:
                fallback_reason = (getattr(error, "stderr", "") or str(error))[-400:]
                output.unlink(missing_ok=True)
        else:
            fallback_reason = f"Multispectral checkpoint missing: {checkpoint}"

    checkpoint = Path(os.getenv("CUSTOM_EARTHDIAL_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not (checkpoint / "run.json").is_file():
        raise RoverError(f"RGB caption checkpoint is missing: {checkpoint}")

    command = [
        str(python), str(CUSTOM / "caption_images.py"),
        "--adapter", str(checkpoint),
        "--before", str(before),
        "--after", str(after),
        "--before-datetime", str(scene.get("before", {}).get("datetime", "")),
        "--after-datetime", str(scene.get("after", {}).get("datetime", "")),
        "--output", str(output),
    ]
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=600,
            check=True,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise RoverError(f"Caption generation failed: {detail[-1000:]}") from error
    if not output.is_file():
        raise RoverError(f"Caption generator did not save {output}: {completed.stdout[-500:]}")
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
        result["mode"] = "rgb_fallback" if fallback_reason else "rgb"
        result["checkpoint"] = str(checkpoint)
        if fallback_reason:
            result["fallback_reason"] = fallback_reason
        from utils import save_json
        save_json("build/captions.json", result)
        return result
    except (OSError, json.JSONDecodeError) as error:
        raise RoverError(f"Could not read saved captions: {error}") from error
