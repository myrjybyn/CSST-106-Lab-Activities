"""LAB 07: threshold, region, edge, and semantic image segmentation."""
import math

import cv2
import numpy as np

from .lab_01 import encode

MAX_PIXELS = 12_000_000
_semantic_model = None


def _decode(contents, name="image"):
    if not contents:
        raise ValueError(f"Upload a non-empty {name}.")
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read the {name}. Use a valid PNG, JPEG, BMP, or WebP file.")
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise ValueError("Use an image with at most 12 million pixels.")
    return image


def _number(options, key, default, low, high, integer=False):
    value = options.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number between {low} and {high}.")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a number between {low} and {high}.") from error
    if not math.isfinite(value) or not low <= value <= high or (integer and int(value) != value):
        raise ValueError(f"{key} must be between {low} and {high}.")
    return int(value) if integer else value


def _iou(mask, truth):
    if truth is None:
        return None
    intersection = int(np.count_nonzero((mask > 0) & (truth > 0)))
    union = int(np.count_nonzero((mask > 0) | (truth > 0)))
    return 1.0 if union == 0 else round(intersection / union, 4)


def _image(image):
    return {"url": encode(image), "shape": list(image.shape)}


def process_image(contents, options, ground_truth_contents=None):
    image = _decode(contents)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    truth = None
    if ground_truth_contents:
        truth_image = cv2.imdecode(np.frombuffer(ground_truth_contents, np.uint8), cv2.IMREAD_GRAYSCALE)
        if truth_image is None:
            raise ValueError("Could not read the ground-truth mask as an image.")
        if truth_image.shape != (height, width):
            raise ValueError("The ground-truth mask must have the same width and height as the input image.")
        _, truth = cv2.threshold(truth_image, 0, 255, cv2.THRESH_BINARY)

    polarity = options.get("polarity", "light")
    if polarity not in ("light", "dark"):
        raise ValueError("Foreground polarity must be light or dark.")
    threshold_mode = cv2.THRESH_BINARY if polarity == "light" else cv2.THRESH_BINARY_INV
    otsu_value, otsu = cv2.threshold(gray, 0, 255, threshold_mode | cv2.THRESH_OTSU)

    block = _number(options, "block_size", 21, 3, 51, integer=True)
    if block % 2 == 0:
        raise ValueError("Adaptive block size must be an odd number from 3 to 51.")
    adaptive_c = _number(options, "adaptive_c", 5, -20, 20)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     threshold_mode, block, adaptive_c)

    seed_x = _number(options, "seed_x", width // 2, 0, width - 1, integer=True)
    seed_y = _number(options, "seed_y", height // 2, 0, height - 1, integer=True)
    tolerance = _number(options, "tolerance", 25, 0, 100, integer=True)
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    flood_work = image.copy()
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(flood_work, flood_mask, (seed_x, seed_y), 0,
                  loDiff=(tolerance,) * 3, upDiff=(tolerance,) * 3,
                  flags=flags)
    region = flood_mask[1:-1, 1:-1].copy()
    region_overlay = image.copy()
    region_overlay[region > 0] = (0.35 * region_overlay[region > 0] +
                                  0.65 * np.array([210, 90, 180])).astype(np.uint8)
    cv2.drawMarker(region_overlay, (seed_x, seed_y), (0, 255, 255), cv2.MARKER_CROSS, 17, 2)

    low = _number(options, "canny_low", 80, 0, 254, integer=True)
    high = _number(options, "canny_high", 160, 1, 255, integer=True)
    if low >= high:
        raise ValueError("The lower Canny threshold must be smaller than the upper threshold.")
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), low, high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(gray)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)
    contour_view = image.copy()
    cv2.drawContours(contour_view, contours, -1, (50, 220, 120), 2)

    fraction = _number(options, "foreground_fraction", 0.35, 0.1, 0.9)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance, fraction * float(distance.max()), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    count, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(image.copy(), markers)
    watershed_mask = np.where(markers > 1, 255, 0).astype(np.uint8)
    watershed_view = np.zeros_like(image)
    watershed_view[markers == -1] = (40, 40, 240)
    palette = np.array([[80, 180, 250], [80, 220, 130], [230, 120, 190],
                        [210, 190, 60], [180, 100, 240], [70, 210, 220]], dtype=np.uint8)
    for label in np.unique(markers):
        if label > 1:
            watershed_view[markers == label] = palette[(int(label) - 2) % len(palette)]

    masks = {"otsu": otsu, "adaptive": adaptive, "region": region,
             "contours": contour_mask, "watershed": watershed_mask}
    metrics = []
    for name, mask in masks.items():
        metrics.append({"method": name, "foreground_pixels": int(np.count_nonzero(mask)),
                        "iou": _iou(mask, truth)})
    return {
        "properties": {"width": width, "height": height, "pixels": width * height,
                       "otsu_threshold": round(float(otsu_value), 2),
                       "seed": {"x": seed_x, "y": seed_y}, "region_pixels": int(np.count_nonzero(region)),
                       "contours": len(contours), "watershed_regions": max(0, int(count) - 1)},
        "metrics": metrics,
        "images": {"original": _image(image), "otsu": _image(otsu), "adaptive": _image(adaptive),
                   "region": _image(region_overlay), "region_mask": _image(region),
                   "canny": _image(edges), "contours": _image(contour_view),
                   "watershed": _image(watershed_view)},
    }


def semantic_capabilities():
    try:
        import torch
        import torchvision
        return {"available": True, "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
                "message": "DeepLabV3 with a ResNet-50 backbone is available. Its pretrained weights download on the first run."}
    except Exception as error:
        return {"available": False, "torch": None, "torchvision": None,
                "message": "Semantic segmentation needs the PyTorch and torchvision dependencies listed in backend/requirements.txt. " + str(error)}


def semantic_segment(contents):
    global _semantic_model
    image = _decode(contents)
    try:
        import torch
        from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights,
                                                     deeplabv3_resnet50)
        if _semantic_model is None:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            _semantic_model = (deeplabv3_resnet50(weights=weights).eval(),
                               tuple(weights.meta["categories"]))
        model, categories = _semantic_model
        h, w = image.shape[:2]
        scale = min(1.0, 512 / max(h, w))
        resized = cv2.resize(image, (max(32, int(w * scale) // 32 * 32),
                                     max(32, int(h * scale) // 32 * 32)))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.inference_mode():
            labels = model((tensor - mean) / std)["out"].argmax(1)[0].cpu().numpy().astype(np.uint8)
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        palette = np.zeros((len(categories), 3), np.uint8)
        for index in range(len(categories)):
            palette[index] = ((37 * index + 70) % 256, (97 * index + 110) % 256,
                              (173 * index + 45) % 256)
        color_mask = palette[labels]
        overlay = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)
        present, counts = np.unique(labels, return_counts=True)
        return {"available": True, "classes": [{"name": categories[int(i)], "pixels": int(n)}
                                                   for i, n in zip(present, counts)],
                "images": {"semantic": _image(overlay), "semantic_mask": _image(color_mask)}}
    except Exception as error:
        raise RuntimeError("DeepLabV3 could not run. Confirm PyTorch/torchvision are installed and allow the first run to download pretrained weights. Details: " + str(error)) from error
