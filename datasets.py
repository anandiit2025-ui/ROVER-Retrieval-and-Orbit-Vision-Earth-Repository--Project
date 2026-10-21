from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
from pystac_client import Client
from shapely.geometry import box, shape

from analysis import BAD_SCL
from raster_utils import align, read_raster
from utils import RoverError


STAC_URL = "https://earth-search.aws.element84.com/v1"


def asset_info(item, names):
    for name in names:
        if name in item.assets:
            asset = item.assets[name]
            bands = asset.extra_fields.get("raster:bands", [])
            unit = bands[0].get("unit") if bands else None
            url = asset.href
            if url.startswith("s3://"):
                bucket, key = url[5:].split("/", 1)
                url = f"https://{bucket}.s3.amazonaws.com/{key}"
            return {"url": url, "unit": unit}
    return None


def scene_info(item, dataset, bbox):
    wanted = {
        "blue": ["blue", "B02"],
        "green": ["green", "B03"],
        "red": ["red", "B04"],
        "nir": ["nir", "nir08", "B08"],
        "swir": ["swir22", "B12"],
        "rededge1": ["rededge1", "B05"],
        "rededge2": ["rededge2", "B06"],
        "rededge3": ["rededge3", "B07"],
        "swir16": ["swir16", "B11"],
        "visual": ["visual", "TCI"],
        "scl": ["scl", "SCL"],
    }
    if dataset == "sentinel-1":
        wanted = {"vv": ["vv", "VV"]}

    area = box(*bbox)
    footprint = shape(item.geometry) if item.geometry else box(*item.bbox)
    overlap = footprint.intersection(area).area / area.area if area.area else 0

    assets = {}
    for name, aliases in wanted.items():
        data = asset_info(item, aliases)
        if data:
            assets[name] = data

    return {
        "id": item.id,
        "datetime": item.datetime.isoformat(),
        "overlap": round(max(0, min(1, overlap)), 4),
        "cloud": item.properties.get("eo:cloud_cover"),
        "orbit": item.properties.get("sat:relative_orbit"),
        "direction": item.properties.get("sat:orbit_state"),
        "assets": assets,
        "footprint": footprint,
    }


def search(dataset, bbox, start, end, cloud_limit):
    collection = "sentinel-1-grd" if dataset == "sentinel-1" else "sentinel-2-l2a"
    arguments = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{start}/{end}",
        "max_items": 200,
    }
    if dataset == "sentinel-2":
        arguments["query"] = {"eo:cloud_cover": {"lte": cloud_limit}}

    try:
        items = Client.open(STAC_URL).search(**arguments).items()
        return [scene_info(item, dataset, bbox) for item in items]
    except Exception as error:
        raise RoverError(f"Satellite search failed: {error}") from error


def choose_pair(dataset, plan):
    cloud_limit = plan["cloud_cover_max"]
    before = search(
        dataset, plan["bbox"], plan["before_start"], plan["before_end"],
        cloud_limit,
    )
    after = search(
        dataset, plan["bbox"], plan["after_start"], plan["after_end"],
        cloud_limit,
    )
    def inside(scene, first, last):
        day = date.fromisoformat(scene["datetime"][:10])
        return date.fromisoformat(first) <= day <= date.fromisoformat(last)

    before = [scene for scene in before if inside(
        scene, plan["before_start"], plan["before_end"])]
    after = [scene for scene in after if inside(
        scene, plan["after_start"], plan["after_end"])]
    warnings = []

    def covered(items):
        return any(item["overlap"] >= 0.5 and "scl" in item["assets"] for item in items)

    if dataset == "sentinel-2":
        limits = [min(100, cloud_limit + 20), 100]
        for limit in dict.fromkeys(limits):
            retry_before = not covered(before)
            retry_after = not covered(after)
            if not retry_before and not retry_after:
                break
            if retry_before:
                before = search(
                    dataset, plan["bbox"], plan["before_start"],
                    plan["before_end"], limit,
                )
                before = [scene for scene in before if inside(
                    scene, plan["before_start"], plan["before_end"])]
                warnings.append(f"Before-image cloud limit was relaxed to {limit:g}%.")
            if retry_after:
                after = search(
                    dataset, plan["bbox"], plan["after_start"],
                    plan["after_end"], limit,
                )
                after = [scene for scene in after if inside(
                    scene, plan["after_start"], plan["after_end"])]
                warnings.append(f"After-image cloud limit was relaxed to {limit:g}%.")

    if not before or not after:
        raise RoverError(f"No {dataset} images found in both date periods")

    before_target = date.fromisoformat(plan["before_target"])
    after_target = date.fromisoformat(plan["after_target"])

    def distance(scene, target):
        scene_date = date.fromisoformat(scene["datetime"][:10])
        cloud = scene["cloud"] if scene["cloud"] is not None else 100
        return (
            -int(scene["overlap"] >= 0.8),
            abs((scene_date - target).days), cloud, -scene["overlap"],
        )

    before.sort(key=lambda scene: distance(scene, before_target))
    after.sort(key=lambda scene: distance(scene, after_target))

    if dataset == "sentinel-1":
        for after_scene in after:
            for before_scene in before:
                same_orbit = before_scene["orbit"] == after_scene["orbit"]
                same_direction = before_scene["direction"] == after_scene["direction"]
                same_mode = before_scene["id"].split("_")[3] == after_scene["id"].split("_")[3]
                if same_orbit and same_direction and same_mode:
                    return {"before": before_scene, "after": after_scene, "warnings": []}
        for after_scene in after:
            for before_scene in before:
                if before_scene["id"].split("_")[3] == after_scene["id"].split("_")[3]:
                    warning = "No matching radar orbit was found; the closest images were used."
                    return {"before": before_scene, "after": after_scene,
                            "warnings": [warning]}
        raise RoverError("No radar pair has the same polarization mode")

    before = [scene for scene in before if scene["overlap"] >= 0.5 and "scl" in scene["assets"]][:3]
    after = [scene for scene in after if scene["overlap"] >= 0.5 and "scl" in scene["assets"]][:3]
    if not before or not after:
        raise RoverError("The optical images do not cover at least half the location")

    area = box(*plan["bbox"])
    options = []
    for i, first in enumerate(before):
        for j, second in enumerate(after):
            common = first["footprint"].intersection(second["footprint"]).intersection(area).area / area.area
            if common >= 0.5:
                options.append((common, i, j))
    if not options:
        raise RoverError("No optical pair covers the same half of the location")

    def days(i, j):
        first = date.fromisoformat(before[i]["datetime"][:10])
        second = date.fromisoformat(after[j]["datetime"][:10])
        return abs((first - before_target).days) + abs((second - after_target).days)

    near = [option for option in options if days(option[1], option[2]) <= 60]
    options = near or options
    fallback = min(
        options,
        key=lambda item: (item[0] < 0.8, days(item[1], item[2]), -item[0]),
    )
    selected = fallback

    try:
        def read_scl(scene):
            return read_raster(scene["assets"]["scl"]["url"], plan["bbox"], True, 30000)

        before_ids = sorted({i for _, i, _ in options})
        after_ids = sorted({j for _, _, j in options})
        scenes = [before[i] for i in before_ids] + [after[j] for j in after_ids]
        with ThreadPoolExecutor(max_workers=4) as pool:
            data = list(pool.map(read_scl, scenes))
        before_data = dict(zip(before_ids, data[:len(before_ids)]))
        after_data = dict(zip(after_ids, data[len(before_ids):]))

        scores = []
        for common, i, j in options:
            first = align(before_data[i], after_data[j], True)["array"]
            second = after_data[j]["array"]
            clear_first = np.isfinite(first) & ~np.isin(np.rint(first), list(BAD_SCL))
            clear_second = np.isfinite(second) & ~np.isin(np.rint(second), list(BAD_SCL))
            clear = float(np.mean(clear_first & clear_second)) * common
            scores.append((clear, -days(i, j), common, i, j))
        _, _, common, i, j = max(scores)
        selected = (common, i, j)
    except RoverError:
        warnings.append("Local cloud check failed; scene metadata was used.")

    common, i, j = selected
    if common < 0.8:
        warnings.append(
            f"The optical pair shares {common * 100:.1f}% of the requested area; "
            "results describe only that area."
        )
    return {"before": before[i], "after": after[j],
            "common_overlap": round(common, 4), "warnings": warnings}
