from concurrent.futures import ThreadPoolExecutor

import numpy as np

from raster_utils import (
    align, mask_area_km2, read_raster, write_raster,
)
from utils import RoverError, safe_number


BAD_SCL = {0, 1, 3, 8, 9, 10, 11}


def difference_index(first, second):
    total = first + second
    output = np.full(first.shape, np.nan, dtype="float32")
    valid = np.isfinite(first) & np.isfinite(second) & (abs(total) > 0.000001)
    output[valid] = (first[valid] - second[valid]) / total[valid]
    return output


def read_scene(scene, bands, bbox, reference=None):
    for band in bands:
        if band not in scene["assets"]:
            raise RoverError(f"Scene {scene['id']} has no {band} band")
        print(f"  Reading {scene['id']} / {band}")

    def read(band):
        url = scene["assets"][band]["url"]
        try:
            return band, read_raster(url, bbox, band == "scl")
        except RoverError:
            return band, read_raster(url, bbox, band == "scl")

    with ThreadPoolExecutor(max_workers=min(4, len(bands))) as pool:
        rasters = dict(pool.map(read, bands))

    result = {}
    grid = reference or rasters[bands[0]]
    for band in bands:
        raster = rasters[band]
        result[band] = raster if raster is grid else align(
            raster, grid, band == "scl"
        )
    return result, grid


def valid_optical(scene, bands, remove_water=False):
    classes = np.rint(scene["scl"]["array"])
    valid = ~np.isin(classes, list(BAD_SCL))
    if remove_water:
        valid &= classes != 6
    for band in bands:
        valid &= np.isfinite(scene[band]["array"])
    return valid


def save_mask(name, mask, valid, reference):
    values = np.where(valid, mask.astype("uint8"), 255)
    return write_raster(f"{name}.tif", values, reference, byte=True)


def analyze_sentinel2(pair, bbox, event):
    event_bands = {
        "flood": ["green", "nir"],
        "water": ["green", "nir"],
        "vegetation": ["nir", "red"],
        "wildfire": ["nir", "swir"],
    }[event]
    rgb = ["red", "green", "blue"]
    after_bands = list(dict.fromkeys(rgb + event_bands + ["scl"]))
    before_bands = list(dict.fromkeys(rgb + event_bands + ["scl"]))

    after, grid = read_scene(pair["after"], after_bands, bbox)
    before, _ = read_scene(pair["before"], before_bands, bbox, grid)
    remove_water = event in {"vegetation", "wildfire"}
    valid_before = valid_optical(before, event_bands, remove_water)
    valid_after = valid_optical(after, event_bands, remove_water)
    valid = valid_before & valid_after
    if not np.any(valid):
        raise RoverError("No clear pixels are available on both dates")

    before_rgb = np.stack([before[name]["array"] for name in rgb])
    after_rgb = np.stack([after[name]["array"] for name in rgb])
    rasters = {}
    for band in after_bands:
        for period, scene in (("before", before), ("after", after)):
            name = f"s2_{period}_{band}"
            rasters[name] = write_raster(
                f"{name}.tif", scene[band]["array"], grid
            )
    rasters["s2_before_rgb"] = write_raster("s2_before_rgb.tif", before_rgb, grid)
    rasters["s2_after_rgb"] = write_raster("s2_after_rgb.tif", after_rgb, grid)
    measurements = {}
    warnings = []

    if event in {"flood", "water"}:
        before_index = np.where(
            valid_before, difference_index(before["green"]["array"], before["nir"]["array"]),
            np.nan,
        )
        after_index = np.where(
            valid_after, difference_index(after["green"]["array"], after["nir"]["array"]),
            np.nan,
        )
        before_water = valid & (before_index > 0)
        after_water = valid & (after_index > 0)
        new_water = after_water & ~before_water
        lost_water = before_water & ~after_water
        rasters["ndwi_before"] = write_raster("ndwi_before.tif", before_index, grid)
        rasters["ndwi_after"] = write_raster("ndwi_after.tif", after_index, grid)
        rasters["ndwi_change"] = write_raster(
            "ndwi_change.tif", np.where(valid, after_index - before_index, np.nan), grid
        )
        rasters["new_water_mask"] = save_mask("new_water_mask", new_water, valid, grid)
        measurements["new_water_area_km2"] = safe_number(mask_area_km2(new_water, grid))
        measurements["lost_water_area_km2"] = safe_number(mask_area_km2(lost_water, grid))
        measurements["water_before_km2"] = safe_number(mask_area_km2(before_water, grid))
        measurements["water_after_km2"] = safe_number(mask_area_km2(after_water, grid))
        measurements["water_change_km2"] = safe_number(
            measurements["water_after_km2"] - measurements["water_before_km2"]
        )
        warnings.append("The water mask is an estimate, not ground truth.")

    elif event == "vegetation":
        before_index = np.where(
            valid_before, difference_index(before["nir"]["array"], before["red"]["array"]),
            np.nan,
        )
        after_index = np.where(
            valid_after, difference_index(after["nir"]["array"], after["red"]["array"]),
            np.nan,
        )
        change = np.where(valid, after_index - before_index, np.nan)
        rasters["ndvi_before"] = write_raster("ndvi_before.tif", before_index, grid)
        rasters["ndvi_after"] = write_raster("ndvi_after.tif", after_index, grid)
        rasters["vegetation_change"] = write_raster("vegetation_change.tif", change, grid)
        measurements["mean_vegetation_change"] = safe_number(np.nanmean(change))

    else:
        before_index = np.where(
            valid_before, difference_index(before["nir"]["array"], before["swir"]["array"]),
            np.nan,
        )
        after_index = np.where(
            valid_after, difference_index(after["nir"]["array"], after["swir"]["array"]),
            np.nan,
        )
        change = np.where(valid, before_index - after_index, np.nan)
        rasters["nbr_before"] = write_raster("nbr_before.tif", before_index, grid)
        rasters["nbr_after"] = write_raster("nbr_after.tif", after_index, grid)
        rasters["burn_change"] = write_raster("burn_change.tif", change, grid)
        measurements["mean_burn_change"] = safe_number(np.nanmean(change))
        warnings.append("The burn map shows spectral change, not confirmed fire damage.")

    quality = {
        "cloud_before": pair["before"]["cloud"],
        "cloud_after": pair["after"]["cloud"],
        "overlap_before": pair["before"]["overlap"],
        "overlap_after": pair["after"]["overlap"],
        "common_overlap": pair.get("common_overlap"),
        "valid_pixels": int(np.count_nonzero(valid)),
        "valid_percent": safe_number(np.count_nonzero(valid) / valid.size * 100),
        "downsampled": grid["downsampled"],
    }
    inputs = ["s2_before_rgb", "s2_after_rgb"]
    if event in {"flood", "water"}:
        inputs += ["ndwi_before", "ndwi_after"]
    elif event == "vegetation":
        inputs += ["ndvi_before", "ndvi_after"]
    else:
        inputs += ["nbr_before", "nbr_after"]
    earthdial_inputs = [
        {
            "path": rasters[name],
            "label": name.replace("_", " "),
            "modality": "Sentinel-2 optical TIFF",
            "datetime": (pair["before"]["datetime"] if "before" in name
                         else pair["after"]["datetime"]),
        }
        for name in inputs
    ]
    return {"measurements": measurements, "quality": quality,
            "rasters": rasters, "warnings": warnings,
            "earthdial_inputs": earthdial_inputs}


def to_db(values, unit):
    if unit and "db" in unit.lower():
        return values
    output = np.full(values.shape, np.nan, dtype="float32")
    valid = np.isfinite(values) & (values > 0)
    output[valid] = 20 * np.log10(values[valid])
    return output


def analyze_sentinel1(pair, bbox, event):
    before, _ = read_scene(pair["before"], ["vv"], bbox)
    after, grid = read_scene(pair["after"], ["vv"], bbox)
    before_vv = align(before["vv"], grid)["array"]
    after_vv = after["vv"]["array"]
    before_db = to_db(before_vv, pair["before"]["assets"]["vv"]["unit"])
    after_db = to_db(after_vv, pair["after"]["assets"]["vv"]["unit"])
    valid = np.isfinite(before_db) & np.isfinite(after_db)
    if not np.any(valid):
        raise RoverError("The radar images have no shared valid pixels")

    change = after_db - before_db
    rasters = {
        "s1_before_vv": write_raster("s1_before_vv.tif", before_vv, grid),
        "s1_after_vv": write_raster("s1_after_vv.tif", after_vv, grid),
        "s1_before": write_raster("s1_before.tif", before_db, grid),
        "s1_after": write_raster("s1_after.tif", after_db, grid),
        "radar_change": write_raster("radar_change.tif", change, grid),
    }
    measurements = {}
    warnings = []

    if event in {"flood", "water"}:
        low_after = np.nanpercentile(after_db[valid], 20)
        large_drop = min(np.nanpercentile(change[valid], 20), -1)
        already_dark = np.nanpercentile(before_db[valid], 15)
        detected = valid & (after_db <= low_after) & (change <= large_drop)
        detected &= before_db > already_dark
        name = "flood" if event == "flood" else "water"
        rasters[f"radar_{name}_mask"] = save_mask(
            f"radar_{name}_mask", detected, valid, grid
        )
        measurements[f"radar_{name}_area_km2"] = safe_number(
            mask_area_km2(detected, grid)
        )
        warnings.append(
            f"The radar {name} area is a candidate estimate, not ground truth."
        )

    quality = {
        "overlap_before": pair["before"]["overlap"],
        "overlap_after": pair["after"]["overlap"],
        "valid_pixels": int(np.count_nonzero(valid)),
        "downsampled": grid["downsampled"],
    }
    earthdial_inputs = [
        {
            "path": rasters[name],
            "label": name.replace("_", " "),
            "modality": "Sentinel-1 SAR VV TIFF",
            "datetime": (pair["before"]["datetime"] if "_before" in name
                         else pair["after"]["datetime"] if "_after" in name
                         else f"{pair['before']['datetime']} to {pair['after']['datetime']}"),
        }
        for name in ("s1_before", "s1_after")
    ]
    return {"measurements": measurements, "quality": quality,
            "rasters": rasters, "warnings": warnings,
            "earthdial_inputs": earthdial_inputs}
