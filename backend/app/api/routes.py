"""HTTP routes; computer vision operations live in the lab modules."""
import json
import re
from pathlib import Path

from flask import Blueprint, abort, current_app, jsonify, request, send_file, url_for

from ..labs.lab_01 import process_image
from ..labs.lab_02 import process_image as process_edges
from ..labs.lab_03 import process_image as process_histograms
from ..labs.lab_04 import capabilities, process_images

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
