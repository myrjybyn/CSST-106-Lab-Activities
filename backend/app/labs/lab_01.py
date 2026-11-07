"""Lab 1: image representation, pixel access, and basic OpenCV operations."""
import base64
import math

import cv2
import numpy as np

MAX_PIXELS = 12_000_000


def integer(options, key, default, minimum, maximum):
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be an integer.")
    if not math.isfinite(value) or int(value) != value or not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}.")
    return int(value)


def encode(image):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the processed image.")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def process_image(contents, options):
    if not contents:
        raise ValueError("The uploaded file is empty.")
    try:
        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError("Cannot decode this image. Use a smaller, valid image file.") from error
    if image is None:
        raise ValueError("Cannot read this image. Use a valid PNG, JPEG, BMP, or WebP file.")
    height, width, channels = image.shape
    if height * width > MAX_PIXELS:
        raise ValueError("Use an image with at most 12 million pixels.")

    x = integer(options, "pixel_x", 0, 0, width - 1)
    y = integer(options, "pixel_y", 0, 0, height - 1)
    bgr = [integer(options, key, default, 0, 255) for key, default in (("blue", 0), ("green", 0), ("red", 255))]
    crop_x = integer(options, "crop_x", 0, 0, width - 1)
    crop_y = integer(options, "crop_y", 0, 0, height - 1)
    crop_w = integer(options, "crop_width", max(1, width // 2), 1, width - crop_x)
    crop_h = integer(options, "crop_height", max(1, height // 2), 1, height - crop_y)
    resize_w = integer(options, "resize_width", max(1, width // 2), 1, 4096)
    resize_h = integer(options, "resize_height", max(1, height // 2), 1, 4096)
    if resize_w * resize_h > MAX_PIXELS:
        raise ValueError("Resized output must have at most 12 million pixels.")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    modified = image.copy()
    before = image[y, x].tolist()
    modified[y, x] = bgr
    # A nearest-neighbor patch makes a single changed pixel visible in the UI.
    left, top = max(0, x - 4), max(0, y - 4)
    patch = modified[top:min(height, y + 5), left:min(width, x + 5)]
    zoom = cv2.resize(patch, (180, 180), interpolation=cv2.INTER_NEAREST)
    cropped = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    interpolation = cv2.INTER_AREA if resize_w <= width and resize_h <= height else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resize_w, resize_h), interpolation=interpolation)
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    outputs = {"original": image, "grayscale": grayscale, "pixel_modified": modified,
               "pixel_zoom": zoom, "cropped": cropped, "resized": resized, "rotated": rotated}
    return {
        "properties": {"width": width, "height": height, "channels": channels,
                       "shape": list(image.shape), "dtype": str(image.dtype),
                       "pixels": width * height, "decoded_bytes": image.nbytes,
                       "color_order": "BGR", "grayscale_shape": list(grayscale.shape)},
        "pixel": {"x": x, "y": y, "before_bgr": before, "after_bgr": bgr,
                  "grayscale_intensity": int(grayscale[y, x])},
        "operations": {"crop": {"x": crop_x, "y": crop_y, "width": crop_w, "height": crop_h},
                       "resize": {"width": resize_w, "height": resize_h},
                       "rotation": "90 degrees clockwise"},
        "images": {name: {"url": encode(output), "shape": list(output.shape)} for name, output in outputs.items()},
    }
