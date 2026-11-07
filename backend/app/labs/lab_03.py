"""LAB 03: histogram equalization, CLAHE, and linear contrast stretching."""
import base64
import io
import math

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .lab_01 import MAX_PIXELS, encode, integer


def histogram_plot(counts, maximum, title, channel):
    figure = Figure(figsize=(6, 2.6), dpi=120, facecolor="#fffaf3")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.set_facecolor("#fffaf3")
    axes.bar(np.arange(256), counts, width=1, color="#76519d")
    axes.set(xlim=(0, 255), ylim=(0, max(1, maximum) * 1.05),
             xlabel=f"{channel} intensity (0-255)", ylabel="Pixel count", title=title)
    axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 4))
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def process_image(contents, options):
    if not contents:
        raise ValueError("The uploaded file is empty.")
    try:
        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError("Cannot decode this image. Use a smaller, valid image file.") from error
    if image is None:
        raise ValueError("Cannot read this image. Use a valid PNG, JPEG, BMP, or WebP file.")
    height, width = image.shape[:2]
    if height * width > MAX_PIXELS:
        raise ValueError("Use an image with at most 12 million pixels.")
    mode = options.get("mode", "grayscale")
    if mode not in ("grayscale", "color"):
        raise ValueError("Choose grayscale or color mode.")
    clip = options.get("clip_limit", 2.0)
    if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not math.isfinite(clip) or not 0.1 <= clip <= 40:
        raise ValueError("CLAHE clip limit must be between 0.1 and 40.")
    grid = integer(options, "tile_grid", 8, 1, 32)
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) if mode == "color" else None
    original = yuv[:, :, 0] if yuv is not None else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    minimum, maximum = int(original.min()), int(original.max())
    stretched = original.copy()
    if maximum > minimum:
        stretched = np.rint((original.astype(np.float64) - minimum) * 255 / (maximum - minimum)).clip(0, 255).astype(np.uint8)
    planes = {
        "original": original,
        "equalized": cv2.equalizeHist(original),
        "clahe": cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(grid, grid)).apply(original),
        "stretched": stretched,
    }
    histograms = {key: cv2.calcHist([plane], [0], None, [256], [0, 256]).ravel().astype(int).tolist()
                  for key, plane in planes.items()}
    histogram_max = max(max(counts) for counts in histograms.values())
    channel = "Y luminance" if mode == "color" else "Grayscale"
    titles = {"original": "Original", "equalized": "Global equalization", "clahe": "CLAHE", "stretched": "Linear contrast stretching"}
    results = {}
    for key, plane in planes.items():
        output = plane
        if yuv is not None:
            changed = yuv.copy()
            changed[:, :, 0] = plane
            output = image if key == "original" else cv2.cvtColor(changed, cv2.COLOR_YUV2BGR)
        results[key] = {"url": encode(output), "shape": list(output.shape),
                        "histogram": histograms[key],
                        "histogram_url": histogram_plot(histograms[key], histogram_max, titles[key], channel),
                        "statistics": {"min": int(plane.min()), "max": int(plane.max()),
                                       "mean": round(float(plane.mean()), 2), "std": round(float(plane.std()), 2)}}
    return {"properties": {"width": width, "height": height, "channel": channel},
            "parameters": {"mode": mode, "clip_limit": float(clip), "tile_grid": grid,
                           "stretch_min": minimum, "stretch_max": maximum}, "images": results}
