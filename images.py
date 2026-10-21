from pathlib import Path

import numpy as np
import torch
from PIL import Image


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def open_image(path):
    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return Image.open(path).convert("RGB")

    import rasterio

    with rasterio.open(path) as source:
        data = source.read([1, 2, 3], masked=True).astype("float32").filled(np.nan)
    valid = np.all(np.isfinite(data), axis=0)
    data = np.power(np.clip(data / 3000, 0, 1), 0.8)
    data[:, ~valid] = 0.85
    return Image.fromarray(np.moveaxis(np.rint(data * 255).astype("uint8"), 0, -1))


def prepare_pair(before, after, size=448, max_tiles=3, mask=None):
    before = before.convert("RGB")
    after = after.convert("RGB")
    if after.size != before.size:
        after = after.resize(before.size, Image.Resampling.BICUBIC)

    width, height = before.size
    boxes = [(0, 0, width, height)]
    if max_tiles > 1 and width >= size // 2 and height >= size // 2:
        candidates = []
        first = np.asarray(before, dtype="float32")
        second = np.asarray(after, dtype="float32")
        change = np.mean(np.abs(first - second), axis=2)
        if mask is not None:
            change = np.asarray(mask.resize(before.size), dtype="float32")
        for row in range(2):
            for col in range(2):
                box = (
                    col * width // 2, row * height // 2,
                    (col + 1) * width // 2, (row + 1) * height // 2,
                )
                left, top, right, bottom = box
                score = float(change[top:bottom, left:right].mean())
                candidates.append((score, row, col, box))
        candidates.sort(reverse=True)
        boxes += [item[3] for item in candidates[:max_tiles - 1]]

    def to_tensor(image, box):
        image = image.crop(box).resize((size, size), Image.Resampling.BICUBIC)
        data = torch.from_numpy(np.asarray(image, dtype="float32").copy())
        data = data.permute(2, 0, 1) / 255
        return (data - MEAN) / STD

    first = torch.stack([to_tensor(before, box) for box in boxes])
    second = torch.stack([to_tensor(after, box) for box in boxes])
    return first, second, boxes
