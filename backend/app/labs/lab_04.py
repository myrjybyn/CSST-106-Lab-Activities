"""LAB 04: local features and two-image descriptor matching.

SURF is used only when the installed OpenCV build can create it. Alternatives
retain their own names; unavailable algorithms never produce substitute output.
"""
import math
from time import perf_counter

import cv2
import numpy as np

from .lab_01 import encode, integer

MAX_FEATURE_PIXELS = 4_000_000
MAX_DESCRIPTORS = 1000


def make_detector(name, threshold=400):
    if name in ("SURF", "U-SURF"):
        return cv2.xfeatures2d.SURF_create(hessianThreshold=threshold, upright=name == "U-SURF")
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=MAX_DESCRIPTORS)
    return cv2.ORB_create(nfeatures=MAX_DESCRIPTORS)


def capabilities():
    result = []
    for name in ("SURF", "U-SURF", "SIFT", "ORB"):
        try:
            make_detector(name)
            result.append({"name": name, "available": True, "reason": ""})
        except (AttributeError, cv2.error):
            reason = ("SURF requires an OpenCV contrib build with OPENCV_ENABLE_NONFREE=ON. The installed build cannot create it."
                      if name in ("SURF", "U-SURF") else f"{name} is unavailable in the installed OpenCV build.")
            result.append({"name": name, "available": False, "reason": reason})
    return result


def decode(contents):
    if not contents:
        raise ValueError("Both image files must contain valid image data.")
    try:
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    except cv2.error as error:
        raise ValueError("Cannot decode one of the images.") from error
    if image is None:
        raise ValueError("Cannot read one of the images. Use PNG, JPEG, BMP or WebP.")
    if image.size > MAX_FEATURE_PIXELS:
        raise ValueError("Each LAB 04 image must have at most 4 million pixels.")
    return image


def extract(detector, image):
    start = perf_counter()
    points, descriptors = detector.detectAndCompute(image, None)
    elapsed = (perf_counter() - start) * 1000
    count = len(points)
    # Bound brute-force matching cost consistently for all algorithms.
    indices = sorted(range(count), key=lambda index: points[index].response, reverse=True)[:MAX_DESCRIPTORS]
    selected = [points[index] for index in indices]
    descriptors = descriptors[indices] if descriptors is not None and indices else None
    return selected, descriptors, count, round(elapsed, 2)


def process_images(first, second, options):
    image_a, image_b = decode(first), decode(second)
    threshold = integer(options, "hessian_threshold", 400, 1, 10000)
    ratio = options.get("ratio", 0.75)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0.1 <= ratio <= 0.95:
        raise ValueError("Match ratio must be between 0.1 and 0.95.")
    results = []
    for capability in capabilities():
        name = capability["name"]
        if not capability["available"]:
            results.append(capability)
            continue
        try:
            detector = make_detector(name, threshold)
            points_a, desc_a, count_a, time_a = extract(detector, image_a)
            points_b, desc_b, count_b, time_b = extract(detector, image_b)
            norm = cv2.NORM_HAMMING if name == "ORB" else cv2.NORM_L2
            good = []
            match_start = perf_counter()
            if desc_a is not None and desc_b is not None and len(desc_b) >= 2:
                pairs = cv2.BFMatcher(norm).knnMatch(desc_a, desc_b, k=2)
                good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
                good.sort(key=lambda match: match.distance)
            match_ms = round((perf_counter() - match_start) * 1000, 2)
            drawn_a = cv2.drawKeypoints(image_a, points_a, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            drawn_b = cv2.drawKeypoints(image_b, points_b, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            matches = cv2.drawMatches(image_a, points_a, image_b, points_b, good[:50], None,
                                     matchColor=(0, 180, 0), singlePointColor=(255, 0, 0),
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            results.append({"name": name, "available": True, "reason": "",
                            "keypoints_a": count_a, "keypoints_b": count_b,
                            "used_a": len(points_a), "used_b": len(points_b),
                            "descriptor_a": list(desc_a.shape) if desc_a is not None else None,
                            "descriptor_b": list(desc_b.shape) if desc_b is not None else None,
                            "descriptor_dtype": str(desc_a.dtype) if desc_a is not None else None,
                            "detect_ms_a": time_a, "detect_ms_b": time_b, "match_ms": match_ms,
                            "matches": len(good), "displayed_matches": min(50, len(good)),
                            "distance": "Hamming" if name == "ORB" else "L2",
                            "images": {"keypoints_a": encode(drawn_a), "keypoints_b": encode(drawn_b), "matches": encode(matches)}})
        except cv2.error:
            results.append({"name": name, "available": False,
                            "reason": "OpenCV could not process these images with this detector. Try larger or different images."})
    return {"parameters": {"hessian_threshold": threshold, "ratio": float(ratio), "descriptor_cap": MAX_DESCRIPTORS},
            "originals": {"a": encode(image_a), "b": encode(image_b)},
            "shapes": {"a": list(image_a.shape), "b": list(image_b.shape)}, "algorithms": results}
