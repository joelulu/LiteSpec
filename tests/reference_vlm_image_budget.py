"""Explicit pixel-area limits for Qwen image processors; no manual resizing."""


def configure_image_budget(processor, min_pixels=None, max_pixels=None):
    image = getattr(processor, "image_processor", None)
    size = getattr(image, "size", None)
    if image is None or size is None:
        raise ValueError("pixel override requires a Qwen image processor with size metadata")
    size = dict(size)
    if "shortest_edge" not in size or "longest_edge" not in size:
        raise ValueError("unsupported image processor size policy; refusing an ineffective pixel override")
    if min_pixels is not None:
        size["shortest_edge"] = int(min_pixels)
    if max_pixels is not None:
        size["longest_edge"] = int(max_pixels)
    if size["shortest_edge"] < 1 or size["longest_edge"] < size["shortest_edge"]:
        raise ValueError("image pixel bounds must be positive and min <= max")
    image.size = size
    return size
