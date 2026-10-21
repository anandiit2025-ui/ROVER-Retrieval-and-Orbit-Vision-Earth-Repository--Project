import os

from utils import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build" / ".matplotlib"))

import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_array(relative_path):
    with rasterio.open(ROOT / relative_path) as source:
        return source.read(masked=True).astype("float32").filled(np.nan)


def rgb_image(array):
    result = np.clip(array / 3000, 0, 1)
    result = np.power(result, 0.8)
    result = np.moveaxis(result, 0, -1)
    result[~np.all(np.isfinite(array), axis=0)] = 0.85
    return result


def save_image(values, path, title, colour_map=None, minimum=None, maximum=None):
    figure, axis = plt.subplots(figsize=(7, 7))
    if colour_map:
        colour_map = matplotlib.colormaps[colour_map].copy()
        colour_map.set_bad("#d9d9d9")
    image = axis.imshow(values, cmap=colour_map, vmin=minimum, vmax=maximum)
    axis.set_title(title)
    axis.set_axis_off()
    if colour_map:
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def create_visuals(rasters, location, event):
    folder = ROOT / "build" / "images"
    folder.mkdir(parents=True, exist_ok=True)

    name = "".join(letter.lower() if letter.isalnum() else "_" for letter in location).strip("_")
    visuals = {}

    for period in ("before", "after"):
        key = f"s2_{period}_rgb"
        if key in rasters:
            path = folder / f"{name}_s2_{period}.png"
            save_image(rgb_image(read_array(rasters[key])), path,
                       f"{location}: Sentinel-2 {period}")
            visuals[f"{period}_image"] = str(path.relative_to(ROOT))

    radar_limits = (None, None)
    if "s1_before" in rasters and "s1_after" in rasters:
        both = np.concatenate([
            read_array(rasters["s1_before"])[0].ravel(),
            read_array(rasters["s1_after"])[0].ravel(),
        ])
        valid = both[np.isfinite(both)]
        if valid.size:
            radar_limits = tuple(float(value) for value in np.percentile(valid, [2, 98]))
    for period in ("before", "after"):
        key = f"s1_{period}"
        if key in rasters:
            path = folder / f"{name}_s1_{period}.png"
            save_image(read_array(rasters[key])[0], path,
                       f"{location}: Sentinel-1 {period}", "gray", *radar_limits)
            relative = str(path.relative_to(ROOT))
            output_key = (f"sentinel1_{period}_image" if f"{period}_image" in visuals
                          else f"{period}_image")
            visuals[output_key] = relative

    result_options = {
        "ndwi_before": ("NDWI before", "BrBG", -1, 1),
        "ndwi_after": ("NDWI after", "BrBG", -1, 1),
        "ndwi_change": ("NDWI change", "BrBG", None, None),
        "ndvi_before": ("NDVI before", "RdYlGn", -1, 1),
        "ndvi_after": ("NDVI after", "RdYlGn", -1, 1),
        "vegetation_change": ("NDVI change", "RdYlGn", None, None),
        "nbr_before": ("Burn index before", "inferno", -1, 1),
        "nbr_after": ("Burn index after", "inferno", -1, 1),
        "burn_change": ("Burn-index change", "inferno", None, None),
        "new_water_mask": ("New water candidate", "Blues", 0, 1),
        "radar_flood_mask": ("Radar flood candidate", "Blues", 0, 1),
        "radar_water_mask": ("Radar water candidate", "Blues", 0, 1),
        "radar_change": ("Radar change", "gray", None, None),
    }
    preferred = {
        "flood": ["ndwi_change", "radar_flood_mask", "radar_change"],
        "water": ["ndwi_change", "radar_water_mask", "radar_change"],
        "vegetation": ["vegetation_change"],
        "wildfire": ["burn_change"],
    }[event]
    primary = next((key for key in preferred if key in rasters), None)
    for key, (title, colour_map, minimum, maximum) in result_options.items():
        if key not in rasters:
            continue
        values = read_array(rasters[key])[0]
        if minimum is None and maximum is None:
            valid = values[np.isfinite(values)]
            if valid.size:
                limit = max(float(np.percentile(np.abs(valid), 98)), 0.02)
                minimum, maximum = -limit, limit
        path = folder / f"{name}_{key}.png"
        save_image(values, path, f"{location}: {title}",
                   colour_map, minimum, maximum)
        visuals["result_image" if key == primary else f"{key}_image"] = str(
            path.relative_to(ROOT)
        )
    return visuals
