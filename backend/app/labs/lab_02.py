"""LAB 02: Gaussian preprocessing, Sobel gradients, and Canny edges."""
import cv2
import numpy as np

from .lab_01 import MAX_PIXELS, encode, integer


def process_image(contents, options):
    if not contents:
        raise ValueError("The uploaded file is empty.")
    try:
        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except cv2.error as error:
        raise ValueError("Cannot decode this image. Use a smaller, valid image file.") from error
    if image is None:
        raise ValueError("Cannot read this image. Use a valid PNG, JPEG, BMP, or WebP file.")
    height, width = image.shape
    if height * width > MAX_PIXELS:
        raise ValueError("Use an image with at most 12 million pixels.")
    lower = integer(options, "lower_threshold", 100, 0, 255)
    upper = integer(options, "upper_threshold", 200, 0, 255)
    if lower >= upper:
        raise ValueError("The lower Canny threshold must be less than the upper threshold.")

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    # Match matplotlib's per-image grayscale display scaling without clipping at 255.
    sobel = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canny = cv2.Canny(blurred, lower, upper)
    images = {"original": image, "sobel": sobel, "canny": canny, "blurred": blurred}
    return {
        "properties": {"width": width, "height": height, "shape": list(image.shape),
                       "canny_edge_pixels": int(np.count_nonzero(canny)),
                       "canny_edge_percent": round(float(np.count_nonzero(canny)) / image.size * 100, 2),
                       "sobel_max_magnitude": round(float(magnitude.max()), 2)},
        "parameters": {"gaussian_kernel": [3, 3], "sobel_kernel": 3,
                       "lower_threshold": lower, "upper_threshold": upper},
        "images": {key: {"url": encode(value), "shape": list(value.shape)} for key, value in images.items()},
    }
