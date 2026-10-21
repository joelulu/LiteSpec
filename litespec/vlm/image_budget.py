"""Materialize bounded images once for serving, validation and training."""
import hashlib
import math
import os
from pathlib import Path


def bounded_size(width, height, *, factor, min_pixels, max_pixels):
    if min(width, height, factor) < 1 or not factor * factor <= min_pixels <= max_pixels:
        raise ValueError("invalid image dimensions/pixel budget")
    if max(width, height) / min(width, height) > 200:
        raise ValueError("image aspect ratio exceeds 200")
    w, h = max(factor, round(width / factor) * factor), max(factor, round(height / factor) * factor)
    if w * h > max_pixels:
        scale = math.sqrt(width * height / max_pixels)
        w, h = max(factor, math.floor(width / scale / factor) * factor), max(factor, math.floor(height / scale / factor) * factor)
    elif w * h < min_pixels:
        scale = math.sqrt(min_pixels / (width * height))
        w, h = math.ceil(width * scale / factor) * factor, math.ceil(height * scale / factor) * factor
    if not min_pixels <= w * h <= max_pixels:
        raise ValueError("aspect ratio and aligned dimensions cannot satisfy pixel budget")
    return w, h


def materialize_image(record, directory, *, factor, min_pixels, max_pixels):
    from PIL import Image, ImageOps
    original = Path(record["image"])
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    key = hashlib.sha256(f"{digest}:{factor}:{min_pixels}:{max_pixels}".encode()).hexdigest()
    path = Path(directory) / (key + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(original) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        dimensions = bounded_size(*image.size, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            image.resize(dimensions, Image.Resampling.BICUBIC).save(temporary, format="PNG")
            os.replace(temporary, path)
    return {**record, "source_image": str(original), "image": str(path.resolve()),
            "split": "val" if int(digest, 16) % 100 < 5 else "train",
            "image_budget": dict(width=dimensions[0], height=dimensions[1], factor=factor,
                                 min_pixels=min_pixels, max_pixels=max_pixels, source_sha256=digest)}
