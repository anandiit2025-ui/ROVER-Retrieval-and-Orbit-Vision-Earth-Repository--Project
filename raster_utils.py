import math
import os

import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from utils import ROOT, RoverError


def read_window(source, bbox, nearest, limit=None):
    bounds = transform_bounds("EPSG:4326", source.crs, *bbox)
    window = from_bounds(*bounds, transform=source.transform)
    window = window.intersection(Window(0, 0, source.width, source.height))
    window = window.round_offsets().round_lengths()

    pixels = window.width * window.height
    limit = limit or int(os.getenv("ROVER_MAX_PIXELS", "500000"))
    scale = min(1, math.sqrt(limit / pixels))
    width = max(1, round(window.width * scale))
    height = max(1, round(window.height * scale))
    method = Resampling.nearest if nearest else Resampling.bilinear
    data = source.read(
        1, window=window, out_shape=(height, width),
        resampling=method, masked=True,
    )
    transform = source.window_transform(window) * Affine.scale(
        window.width / width, window.height / height
    )
    return {
        "array": data.astype("float32").filled(np.nan),
        "transform": transform,
        "crs": source.crs,
        "downsampled": scale < 1,
    }


def read_raster(url, bbox, nearest=False, limit=None):
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(url) as source:
                if source.crs:
                    return read_window(source, bbox, nearest, limit)
                points, crs = source.gcps
                if not points or not crs:
                    raise RoverError("The raster has no usable coordinates")
                with WarpedVRT(source, crs="EPSG:4326") as warped:
                    return read_window(warped, bbox, nearest, limit)
    except Exception as error:
        raise RoverError(f"Could not read satellite pixels: {error}") from error


def align(raster, reference, nearest=False):
    output = np.full(reference["array"].shape, np.nan, dtype="float32")
    method = Resampling.nearest if nearest else Resampling.bilinear
    reproject(
        raster["array"], output,
        src_transform=raster["transform"], src_crs=raster["crs"],
        dst_transform=reference["transform"], dst_crs=reference["crs"],
        src_nodata=np.nan, dst_nodata=np.nan, resampling=method,
    )
    return {**reference, "array": output}


def write_raster(filename, array, reference, byte=False):
    path = ROOT / "build" / "rasters" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array)
    if data.ndim == 2:
        data = data[None, :, :]
    dtype = "uint8" if byte else "float32"
    nodata = 255 if byte else np.nan
    data = data.astype(dtype)

    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=data.shape[0], dtype=dtype, crs=reference["crs"],
        transform=reference["transform"], nodata=nodata, compress="deflate",
    ) as output:
        output.write(data)
    return str(path.relative_to(ROOT))


def mask_area_km2(mask, reference):
    crs = reference["crs"]
    transform = reference["transform"]
    if crs.is_projected:
        pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
        return float(np.count_nonzero(mask) * pixel_area / 1_000_000)

    height, width = mask.shape
    bounds = array_bounds(height, width, transform)
    centre_lon = (bounds[0] + bounds[2]) / 2
    centre_lat = (bounds[1] + bounds[3]) / 2
    zone = int((centre_lon + 180) / 6) + 1
    target = f"EPSG:{32600 + zone if centre_lat >= 0 else 32700 + zone}"
    new_transform, new_width, new_height = calculate_default_transform(
        crs, target, width, height, *bounds
    )
    projected = np.zeros((new_height, new_width), dtype="uint8")
    reproject(mask.astype("uint8"), projected, src_transform=transform,
              src_crs=crs, dst_transform=new_transform, dst_crs=target,
              resampling=Resampling.nearest)
    pixel_area = abs(new_transform.a * new_transform.e)
    return float(np.count_nonzero(projected) * pixel_area / 1_000_000)
