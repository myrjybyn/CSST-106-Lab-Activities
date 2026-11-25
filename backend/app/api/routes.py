"""HTTP routes; computer vision operations live in the lab modules."""
import json
import re
from pathlib import Path

from flask import Blueprint, abort, current_app, jsonify, request, send_file, url_for

from ..labs.lab_01 import process_image
from ..labs.lab_02 import process_image as process_edges
from ..labs.lab_03 import process_image as process_histograms
from ..labs.lab_04 import capabilities, process_images
from ..labs.lab_05 import run_experiment as run_pca_experiment
from ..labs.lab_06 import capabilities as yolo_capabilities, detect as yolo_detect
from ..labs.lab_06 import train as train_yolo, evaluate as evaluate_yolo, checkpoint as yolo_checkpoint
from ..labs.lab_07 import process_image as segment_image, semantic_capabilities, semantic_segment

api = Blueprint("api", __name__)


def report_files():
    root = Path(current_app.config["REPORTS_DIR"]).resolve()
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
        if path.is_file() and path.suffix.lower() == ".pdf" and path.resolve().is_relative_to(root):
            yield path.relative_to(root).as_posix(), path


@api.get("/reports")
def reports():
    items = []
    for relative, path in report_files():
        match = re.search(r"(?:^|[/ _-])lab[ _-]*(\d+)(?=$|[/ _.-])", relative, re.IGNORECASE)
        items.append({"filename": path.name, "path": relative,
                      "lab_number": int(match.group(1)) if match else None,
                      "url": url_for("api.report_pdf", filename=relative)})
    response = jsonify(reports=items)
    response.headers["Cache-Control"] = "no-store"
    return response


@api.get("/reports/files/<path:filename>")
def report_pdf(filename):
    for relative, path in report_files():
        if relative == filename:
            return send_file(path, mimetype="application/pdf", conditional=True)
    abort(404)


@api.get("/labs")
def labs():
    return jsonify(labs=[
        {"id": "lab-01", "number": 1, "title": "Image Manipulation", "status": "available"},
        {"id": "lab-02", "number": 2, "title": "Edge Detection", "status": "available"},
        {"id": "lab-03", "number": 3, "title": "Histogram Equalization and Contrast Adjustment", "status": "available"},
        {"id": "lab-04", "number": 4, "title": "SURF Feature Detection", "status": "available"},
        {"id": "lab-05", "number": 5, "title": "Facial Recognition with PCA (Eigenfaces)", "status": "available"},
        {"id": "lab-06", "number": 6, "title": "Object Detection with YOLOv11", "status": "available"},
        {"id": "lab-07", "number": 7, "title": "Segmentation Technique", "status": "available"},
    ])


@api.post("/labs/1/process")
def lab_one():
    return process_upload(process_image)


@api.post("/labs/2/process")
def lab_two():
    return process_upload(process_edges)


@api.post("/labs/3/process")
def lab_three():
    return process_upload(process_histograms)


@api.get("/labs/4/capabilities")
def feature_capabilities():
    return jsonify(algorithms=capabilities())


@api.post("/labs/4/process")
def lab_four():
    first, second = request.files.get("image_a"), request.files.get("image_b")
    if first is None or second is None or not first.filename or not second.filename:
        return jsonify(error="Choose both images to detect and match features."), 400
    try:
        options = json.loads(request.form.get("options", "{}"))
        if not isinstance(options, dict):
            raise ValueError("Options must be a JSON object.")
        result = process_images(first.read(), second.read(), options)
    except (ValueError, TypeError, OverflowError) as error:
        return jsonify(error=str(error)), 400
    return jsonify(result)


@api.post("/labs/5/run")
def lab_five():
    try:
        return jsonify(run_pca_experiment(current_app.config["SKLEARN_DATA_HOME"]))
    except Exception as error:
        current_app.logger.exception("LAB 05 PCA experiment failed")
        return jsonify(error=("Could not load the LFW dataset or complete PCA. Check your internet connection for the first download, then try again. " + str(error))), 503


@api.get("/labs/6/capabilities")
def yolo_status():
    return jsonify(yolo_capabilities())


@api.post("/labs/7/process")
def lab_seven():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="Choose an image first."), 400
    ground_truth = request.files.get("ground_truth")
    try:
        options = json.loads(request.form.get("options", "{}"))
        if not isinstance(options, dict):
            raise ValueError("Options must be a JSON object.")
        result = segment_image(upload.read(), options,
                               ground_truth.read() if ground_truth and ground_truth.filename else None)
        return jsonify(result)
    except (ValueError, TypeError, OverflowError) as error:
        return jsonify(error=str(error)), 400


@api.get("/labs/7/semantic-capabilities")
def lab_seven_semantic_status():
    return jsonify(semantic_capabilities())


@api.post("/labs/7/semantic")
def lab_seven_semantic():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="Choose an image first."), 400
    try:
        return jsonify(semantic_segment(upload.read()))
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except RuntimeError as error:
        return jsonify(error=str(error)), 503


@api.post("/labs/6/detect")
def yolo_inference():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="Choose an image or capture a webcam snapshot first."), 400
    try:
        confidence = request.form.get("confidence", "0.25")
        image = upload.read()
        model_id = request.form.get("model_id", "pretrained")
        return jsonify(yolo_detect(image, model_id, confidence))
    except (ValueError, TypeError, OverflowError) as error:
        return jsonify(error=str(error)), 400
    except RuntimeError as error:
        return jsonify(error=str(error)), 503


@api.post("/labs/6/train")
def yolo_training():
    upload = request.files.get("dataset")
    if upload is None or not upload.filename:
        return jsonify(error="Upload a ZIP containing data.yaml, images/train, images/val, labels/train, and labels/val."), 400
    try:
        epochs = request.form.get("epochs", "100")
        image_size = request.form.get("image_size", "640")
        result = train_yolo(upload.read(), epochs, image_size)
        return jsonify(result)
    except (ValueError, TypeError, OverflowError) as error:
        return jsonify(error=str(error)), 400
    except RuntimeError as error:
        return jsonify(error=str(error)), 503


@api.post("/labs/6/evaluate")
def yolo_evaluation():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(evaluate_yolo(data.get("model_id", "")))
    except (ValueError, TypeError) as error:
        return jsonify(error=str(error)), 400
    except RuntimeError as error:
        return jsonify(error=str(error)), 503


@api.get("/labs/6/checkpoint/<model_id>")
def yolo_best_checkpoint(model_id):
    try:
        return yolo_checkpoint(model_id)
    except (ValueError, TypeError):
        abort(404)


def process_upload(processor):
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="Choose an image first."), 400
    try:
        options = json.loads(request.form.get("options", "{}"))
        if not isinstance(options, dict):
            raise ValueError("Options must be a JSON object.")
        result = processor(upload.read(), options)
    except (ValueError, TypeError, OverflowError) as error:
        return jsonify(error=str(error)), 400
    return jsonify(result)
