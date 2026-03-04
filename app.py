"""
Flask web application — provides a simple accessible dashboard
for the Knowledge Graph Query System.
"""

from flask import Flask, render_template, request, jsonify, session
from src.config import Config
from src.pipeline import Pipeline

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.secret_key = Config.FLASK_SECRET_KEY

# Global pipeline instance (initialized lazily)
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    """Get or create the pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    """Process a user question and return the answer."""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        pipeline = get_pipeline()
        result = pipeline.process_question(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear_conversation():
    """Clear the conversation history."""
    try:
        pipeline = get_pipeline()
        pipeline.clear_conversation()
        return jsonify({"status": "ok", "message": "Conversation cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache/stats", methods=["GET"])
def cache_stats():
    """Return cache statistics."""
    try:
        pipeline = get_pipeline()
        return jsonify(pipeline.get_cache_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """Clear the query cache."""
    try:
        pipeline = get_pipeline()
        pipeline.clear_cache()
        return jsonify({"status": "ok", "message": "Cache cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/schema", methods=["GET"])
def get_schema():
    """Return the Neo4j database schema."""
    try:
        pipeline = get_pipeline()
        return jsonify({"schema": pipeline.schema})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        pipeline = get_pipeline()
        db_ok = pipeline.db.verify_connectivity()
        return jsonify({
            "status": "healthy" if db_ok else "degraded",
            "database": "connected" if db_ok else "disconnected",
            "cache": pipeline.get_cache_stats(),
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


import atexit

def _shutdown_pipeline():
    """Clean up on app shutdown."""
    global _pipeline
    if _pipeline is not None:
        _pipeline.close()
        _pipeline = None

atexit.register(_shutdown_pipeline)


if __name__ == "__main__":
    errors = Config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        print("\nPlease set the required environment variables in .env")
        exit(1)

    print("Starting Knowledge Graph Query System...")
    app.run(debug=Config.FLASK_DEBUG, host="0.0.0.0", port=5000)
