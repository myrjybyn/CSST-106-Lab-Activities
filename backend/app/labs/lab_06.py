"""LAB 06: YOLOv11 detection, custom dataset training, and validation."""
import io
import math
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from flask import abort, send_file

MAX_IMAGE_PIXELS = 12_000_000
MAX_ARCHIVE_FILES = 5000
MAX_ARCHIVE_EXPANDED = 1024 * 1024 * 1024
_models = {}
_loaded = {}


def _ultralytics():
    try:
        from ultralytics import YOLO
        import ultralytics
    except ImportError as error:
        raise RuntimeError("YOLOv11 needs Ultralytics and PyTorch. From the project root, install project dependencies with `python -m pip install -r backend/requirements.txt`, restart Flask, and try again.") from error
    return YOLO, ultralytics


def capabilities():
    try:
        _, package = _ultralytics()
        return {"available": True, "version": str(package.__version__), "model": "YOLOv11n",
                "message": "Ultralytics is installed. The first pretrained inference downloads yolo11n.pt if it is not cached."}
    except RuntimeError as error:
        return {"available": False, "version": None, "model": "YOLOv11n", "message": str(error)}


def _model(model_id):
    YOLO, _ = _ultralytics()
    if model_id == "pretrained":
        key, source = "pretrained", "yolo11n.pt"
    else:
        item = _models.get(str(model_id))
        if item is None or not item["weights"].is_file():
            raise ValueError("That custom checkpoint is no longer available. Train the model again in this Flask session.")
        key, source = str(model_id), str(item["weights"])
    if key not in _loaded:
        try:
            _loaded[key] = YOLO(source)
        except Exception as error:
            raise RuntimeError(f"Could not load {Path(source).name}. Check the server terminal and try again. Details: {error}") from error
    return _loaded[key]


def detect(contents, model_id="pretrained", confidence="0.25"):
    if len(contents) > 100 * 1024 * 1024:
        raise ValueError("Image must be smaller than 100 MB.")
    try:
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError("Could not decode the image.") from error
    if image is None:
        raise ValueError("Choose a valid PNG, JPEG, BMP, or WebP image.")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Use an image with at most 12 million pixels.")
    try:
        threshold = float(confidence)
    except (TypeError, ValueError) as error:
        raise ValueError("Confidence must be a number between 0.01 and 0.99.") from error
    if not math.isfinite(threshold) or not 0.01 <= threshold <= 0.99:
        raise ValueError("Confidence must be between 0.01 and 0.99.")
    model = _model(model_id)
    start = time.perf_counter()
    try:
        result = model.predict(source=image, conf=threshold, verbose=False)[0]
    except Exception as error:
        raise RuntimeError(f"YOLO inference failed. Details: {error}") from error
    elapsed = (time.perf_counter() - start) * 1000
    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            boxes.append({"class_id": cls_id, "label": str(result.names.get(cls_id, cls_id)),
                          "confidence": round(float(box.conf.item()), 4),
                          "xyxy": [round(float(value), 1) for value in box.xyxy[0].tolist()]})
    annotated = result.plot()
    ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode annotated output.")
    import base64
    return {"width": int(image.shape[1]), "height": int(image.shape[0]),
            "inference_ms": round(elapsed, 2), "model": Path(str(getattr(model, "ckpt_path", model_id))).name,
            "count": len(boxes), "detections": boxes,
            "annotated": "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")}


def _safe_extract(contents):
    if len(contents) > 120 * 1024 * 1024:
        raise ValueError("Dataset ZIP must be smaller than 120 MB.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(contents))
    except (zipfile.BadZipFile, OSError) as error:
        raise ValueError("Upload a valid ZIP dataset archive.") from error
    root = Path(tempfile.mkdtemp(prefix="csst106-yolo-data-")).resolve()
    try:
        entries = archive.infolist()
        if len(entries) > MAX_ARCHIVE_FILES:
            raise ValueError("Dataset archive contains too many files (limit: 5,000).")
        if sum(info.file_size for info in entries) > MAX_ARCHIVE_EXPANDED:
            raise ValueError("Expanded dataset archive must be smaller than 1 GB.")
        for info in entries:
            name = info.filename.replace("\\", "/")
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or ":" in name:
                raise ValueError("Dataset ZIP contains an unsafe file path.")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("Dataset ZIP must not contain symbolic links.")
            destination = (root / Path(*relative.parts)).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Dataset ZIP contains a path outside the archive folder.")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        archive.close()
        yaml_files = list(root.rglob("data.yaml")) + list(root.rglob("data.yml"))
        if len(yaml_files) != 1:
            raise ValueError("ZIP must contain exactly one data.yaml at the dataset root.")
        return root, yaml_files[0]
    except Exception:
        archive.close()
        shutil.rmtree(root, ignore_errors=True)
        raise


def _validate_dataset(contents):
    import yaml
    root, data_yaml = _safe_extract(contents)
    try:
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("data.yaml must contain a YAML mapping.")
        names = data.get("names")
        if isinstance(names, list):
            names = {index: name for index, name in enumerate(names)}
        if not isinstance(names, dict) or not names or any(not isinstance(name, str) or not name.strip() for name in names.values()):
            raise ValueError("data.yaml needs a names list or class-id mapping with non-empty class names.")
        normalized_names = {int(key): value.strip() for key, value in names.items()}
        if sorted(normalized_names) != list(range(len(normalized_names))):
            raise ValueError("Class IDs in names must start at 0 and be consecutive.")
        if "nc" in data and int(data["nc"]) != len(normalized_names):
            raise ValueError("data.yaml nc must match the number of class names.")
        base = (data_yaml.parent / str(data.get("path", ""))).resolve()
        if not base.is_relative_to(root):
            raise ValueError("Dataset path in data.yaml must stay inside the uploaded archive.")
        validated_splits = {}
        split_image_counts = {}
        for split in ("train", "val"):
            entries = data.get(split)
            if isinstance(entries, str):
                entries = [entries]
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"data.yaml must specify {split} images.")
            resolved = []
            for entry in entries:
                directory = (base / str(entry)).resolve()
                if not directory.is_relative_to(root) or not directory.exists():
                    raise ValueError(f"{split} image path is missing or outside the dataset ZIP.")
                resolved.append(directory)
            validated_splits[split] = resolved
        for split, directories in validated_splits.items():
            images = []
            for directory in directories:
                images.extend(directory.rglob("*")) if directory.is_dir() else images.append(directory)
            images = [path for path in images if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}]
            if not images:
                raise ValueError(f"No image files found for the {split} split.")
            split_image_counts[split] = len(images)
            checked_labels = 0
            for image in images:
                parts = list(image.parts)
                if "images" not in parts:
                    raise ValueError("YOLO image paths must use an images/{train,val} folder structure.")
                parts[parts.index("images")] = "labels"
                label = Path(*parts).with_suffix(".txt")
                if not label.exists():
                    raise ValueError(f"Missing YOLO annotation file for {image.name} in the {split} split.")
                for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(f"{label.name}, line {line_no}: expected class_id x_center y_center width height.")
                    try:
                        class_id = int(fields[0])
                        coords = [float(value) for value in fields[1:]]
                    except ValueError as error:
                        raise ValueError(f"{label.name}, line {line_no}: annotation values must be numeric.") from error
                    if not 0 <= class_id < len(normalized_names) or not all(math.isfinite(value) and 0 <= value <= 1 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                        raise ValueError(f"{label.name}, line {line_no}: class IDs and normalized box coordinates are invalid.")
                    checked_labels += 1
            if checked_labels == 0:
                raise ValueError(f"No object annotations found in the {split} labels.")
        data["names"] = normalized_names
        data["nc"] = len(normalized_names)
        data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return root, data_yaml, normalized_names, split_image_counts
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _integer(value, label, minimum, maximum):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer from {minimum} to {maximum}.") from error
    if isinstance(value, bool) or str(number) != str(value).strip() or not minimum <= number <= maximum:
        raise ValueError(f"{label} must be an integer from {minimum} to {maximum}.")
    return number


def train(contents, epochs, image_size):
    YOLO, _ = _ultralytics()
    epochs = _integer(epochs, "Epochs", 1, 100)
    image_size = _integer(image_size, "Image size", 320, 1280)
    if image_size % 32:
        raise ValueError("Image size must be divisible by 32.")
    root, data_yaml, names, split_counts = _validate_dataset(contents)
    model_id = uuid.uuid4().hex
    run_root = Path(tempfile.gettempdir()) / "csst106-yolo-runs" / model_id
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        model = YOLO("yolo11n.pt")
        results = model.train(data=str(data_yaml), epochs=epochs, imgsz=image_size,
                              project=str(run_root), name="training", exist_ok=True,
                              workers=0, verbose=False)
        run_dir = Path(model.trainer.save_dir)
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError("Training finished without producing weights/best.pt.")
        metrics = getattr(results, "results_dict", {}) or {}
        clean_metrics = {str(key): float(value) for key, value in metrics.items()
                         if isinstance(value, (int, float, np.number)) and math.isfinite(float(value))}
        _models[model_id] = {"weights": best, "data_yaml": data_yaml, "dataset_root": root,
                             "names": names, "run_dir": run_dir}
        _loaded[model_id] = YOLO(str(best))
        return {"model_id": model_id, "epochs": epochs, "image_size": image_size,
                "classes": names, "images": split_counts, "metrics": clean_metrics,
                "checkpoint_url": f"/api/labs/6/checkpoint/{model_id}",
                "message": "Training completed. Ultralytics also ran validation during training; use Evaluate to run it again."}
    except Exception as error:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(run_root, ignore_errors=True)
        if isinstance(error, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"YOLO training failed. Check the dataset, available memory, and server output. Details: {error}") from error


def evaluate(model_id):
    item = _models.get(str(model_id))
    if item is None or not item["weights"].is_file():
        raise ValueError("Train a model in this Flask session before evaluating it.")
    model = _loaded.get(str(model_id))
    if model is None:
        YOLO, _ = _ultralytics()
        model = _loaded[str(model_id)] = YOLO(str(item["weights"]))
    try:
        metrics = model.val(data=str(item["data_yaml"]), verbose=False)
    except Exception as error:
        raise RuntimeError(f"Validation failed. Details: {error}") from error
    values = getattr(metrics, "results_dict", {}) or {}
    clean = {str(key): float(value) for key, value in values.items()
             if isinstance(value, (int, float, np.number)) and math.isfinite(float(value))}
    return {"model_id": str(model_id), "metrics": clean}


def checkpoint(model_id):
    item = _models.get(str(model_id))
    if item is None or not item["weights"].is_file():
        abort(404)
    return send_file(item["weights"], as_attachment=True, download_name="best.pt")
