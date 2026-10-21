"""Prepare aligned nine-channel Sentinel-2 inputs for ChatEarthNet captions."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from analysis import BAD_SCL
from raster_utils import align, read_raster
from utils import ROOT, RoverError


# ChatEarthNet's Satlas image order: TCI RGB, then six spectral channels.
SPECTRAL = ("rededge1", "rededge2", "rededge3", "nir", "swir16", "swir")
BANDS = ("B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12")


def prepare_pair(pair, bbox, rasters):
    """Return two aligned 8-bit nine-band GeoTIFFs, or raise for RGB fallback."""
    rgb_path = ROOT / rasters["s2_before_rgb"]
    with rasterio.open(rgb_path) as source:
        reference = {
            "array": np.empty((source.height, source.width), dtype="float32"),
            "transform": source.transform,
            "crs": source.crs,
        }

    stacks = {}
    for role in ("before", "after"):
        scene = pair[role]
        required = (*SPECTRAL, "visual", "scl", "red", "green", "blue")
        missing = [name for name in required if name not in scene["assets"]]
        if missing:
            raise RoverError(f"{role} scene lacks nine-band assets: {', '.join(missing)}")
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                              GDAL_HTTP_CONNECTTIMEOUT="10", GDAL_HTTP_TIMEOUT="25",
                              GDAL_HTTP_MAX_RETRY="1"):
                with rasterio.open(scene["assets"]["visual"]["url"]) as source:
                    if source.count < 3 or source.crs is None:
                        raise RoverError(f"{role} visual asset is not georeferenced RGB")
                    with WarpedVRT(
                        source, crs=reference["crs"], transform=reference["transform"],
                        width=reference["array"].shape[1],
                        height=reference["array"].shape[0],
                        resampling=Resampling.bilinear,
                    ) as warped:
                        rgb = warped.read([1, 2, 3]).astype("float32")
            if not np.isfinite(rgb).all() or rgb.max() > 255 or rgb.min() < 0:
                raise RoverError(f"{role} TCI asset is not 8-bit RGB")
            channels = [rgb]
            for name in SPECTRAL:
                with rasterio.Env(GDAL_HTTP_CONNECTTIMEOUT="10", GDAL_HTTP_TIMEOUT="25",
                                  GDAL_HTTP_MAX_RETRY="1"):
                    raster = align(read_raster(scene["assets"][name]["url"], bbox), reference)
                values = raster["array"]
                if not np.isfinite(values).all():
                    # Nodata is filled only after the clear-pixel check below.
                    values = values.copy()
                channels.append(values[None])
            with rasterio.Env(GDAL_HTTP_CONNECTTIMEOUT="10", GDAL_HTTP_TIMEOUT="25",
                              GDAL_HTTP_MAX_RETRY="1"):
                scl = align(read_raster(scene["assets"]["scl"]["url"], bbox, True), reference, True)["array"]
            stack = np.concatenate(channels, axis=0)
            valid = np.all(np.isfinite(stack), axis=0) & np.isfinite(scl)
            valid &= ~np.isin(np.rint(scl), list(BAD_SCL))
            valid_fraction = float(np.mean(valid))
            if valid_fraction < 0.5:
                raise RoverError(f"{role} nine-band clear coverage is {valid_fraction:.1%}, below 50%")
            # SatlasPretrain's ChatEarthNet source encodes spectral 16-bit bands
            # as uint8 by integer-dividing by 32; TCI remains 8-bit RGB.
            stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
            stack[:3] = np.clip(np.rint(stack[:3]), 0, 255)
            stack[3:] = np.clip(np.floor(stack[3:] / 32), 0, 255)
            stacks[role] = (stack.astype("uint8"), valid_fraction)
        except (OSError, ValueError, rasterio.errors.RasterioError) as error:
            raise RoverError(f"Could not prepare {role} nine-band image: {error}") from error

    output = {}
    for role, (stack, valid_fraction) in stacks.items():
        path = ROOT / "build" / "rasters" / f"s2_{role}_ms9.tif"
        with rasterio.open(
            path, "w", driver="GTiff", width=stack.shape[2], height=stack.shape[1],
            count=9, dtype="uint8", crs=reference["crs"],
            transform=reference["transform"], compress="deflate",
        ) as sink:
            sink.write(stack)
            sink.descriptions = BANDS
        output[role] = {"path": str(path.relative_to(ROOT)), "clear_fraction": valid_fraction}
    return {"images": output, "bands": list(BANDS),
            "preprocessing": "TCI/255; B05-B12 floor(raw/32)/255; aligned to ROVER RGB grid"}
