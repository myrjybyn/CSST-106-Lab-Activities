"""Run from the repository root: python -m backend.app.main."""
from pathlib import Path

from flask import Flask, jsonify, send_from_directory
from werkzeug.exceptions import HTTPException

from .api.routes import api


def create_app():
    frontend = Path(__file__).resolve().parents[2] / "frontend"
    app = Flask(__name__, static_folder=str(frontend / "src"), static_url_path="/assets")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    app.config["REPORTS_DIR"] = frontend.parent / "reports"
    app.register_blueprint(api, url_prefix="/api")

    @app.get("/")
    def index():
        return send_from_directory(frontend / "public", "index.html")

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
