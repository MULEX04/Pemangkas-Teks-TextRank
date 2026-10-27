"""
app.py
------
Flask backend for the TextRank summarizer web application.

Endpoints:
    POST /summarize  -- accepts JSON {text, ratio}, returns summarized result
    GET  /health     -- health check

Run:
    python app.py
Server listens on http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from textrank import textrank_summarize
from scraper import scrape_article

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # Allow requests from the frontend (different origin / file://)

# MAX_WORDS = 500
MAX_WORDS = 3000


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split()) if text.strip() else 0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok", "message": "TextRank server is running."})


@app.route("/scrape", methods=["POST"])
def scrape():
    """
    POST /scrape
    ------------
    Request body (JSON):
        { "url": "https://..." }
    Response (JSON):
        { "text": "...", "title": "...", "domain": "...", "url": "...", "word_count": <int>, "error": null | "..." }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("url"):
        return jsonify({"error": "Field 'url' is required."}), 400

    result = scrape_article(data["url"])
    if result.get("error"):
        return jsonify(result), 422
    return jsonify(result), 200


@app.route("/summarize", methods=["POST"])
def summarize():
    """
    POST /summarize
    ---------------
    Request body (JSON):
        {
            "text":  "<input text>",
            "ratio": 0.30          # float between 0.10 and 0.50
        }

    Response (JSON):
        {
            "summary":        "<extracted summary>",
            "original_sents": <int>,
            "summary_sents":  <int>,
            "word_count":     <int>,   -- original word count
            "scores":         [...],   -- [(sentence, score), ...]
            "error":          null | "<error message>"
        }
    """
    data = request.get_json(silent=True)

    # ── Input validation ────────────────────────────────────────────────────
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required and must not be empty."}), 400

    word_count = count_words(text)
    if word_count > MAX_WORDS:
        return jsonify({
            "error": f"Text exceeds the maximum of {MAX_WORDS} words (got {word_count})."
        }), 400

    # ── Parse ratio ─────────────────────────────────────────────────────────
    try:
        ratio = float(data.get("ratio", 0.30))
    except (TypeError, ValueError):
        return jsonify({"error": "Field 'ratio' must be a number between 0.10 and 0.50."}), 400

    ratio = max(0.10, min(0.50, ratio))

    # ── Run TextRank ─────────────────────────────────────────────────────────
    try:
        result = textrank_summarize(text, ratio=ratio)
    except Exception as exc:
        return jsonify({"error": f"Internal error during summarization: {str(exc)}"}), 500

    if result.get("error"):
        return jsonify({"error": result["error"]}), 422

    return jsonify({
        "summary":        result["summary"],
        "original_sents": result["original_sents"],
        "summary_sents":  result["summary_sents"],
        "word_count":     word_count,
        "scores":         result["scores"],
        "error":          None,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  TextRank Summarizer  |  Flask Backend")
    print("  Listening on:  http://127.0.0.1:5000")
    print("  Health check:  http://127.0.0.1:5000/health")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5000, debug=True)
