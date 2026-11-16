"""Flask web application for MailGuard AI."""
import json
import os
import pickle
import re
import sqlite3
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, render_template, request
from config import DATABASE_DIR, DATABASE_PATH, MAX_MESSAGE_LENGTH, METRICS_PATH, MODEL_PATH, VECTORIZER_PATH

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MESSAGE_LENGTH * 4

def db():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, preview TEXT NOT NULL, label TEXT NOT NULL, confidence REAL NOT NULL, created_at TEXT NOT NULL)")

init_db()
model = vectorizer = metrics = None
try:
    if not all(p.is_file() for p in (MODEL_PATH, VECTORIZER_PATH, METRICS_PATH)):
        raise FileNotFoundError("Run python train.py to create model artifacts.")
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
except Exception as exc:
    # Keep UI usable without model artifacts; frontend errors stay generic.
    app.logger.info("Model unavailable: %s", exc)

def get_history(limit=100):
    with db() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]

@app.get("/")
def home():
    rows = get_history(5000)
    summary = {"total": len(rows), "spam": sum(r["label"] == "SPAM" for r in rows), "ham": sum(r["label"] == "NOT SPAM" for r in rows)}
    return render_template("index.html", summary=summary, metrics=metrics, model_ready=model is not None)

@app.post("/predict")
def predict():
    if model is None or vectorizer is None:
        return jsonify(error="The model is not ready yet. Follow the training steps in README.md."), 503
    data = request.get_json(silent=True) or request.form
    message = data.get("message", "") if data else ""
    if not isinstance(message, str) or not message.strip():
        return jsonify(error="Enter a message before analyzing it."), 400
    message = re.sub(r"\s+", " ", message).strip()
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify(error=f"Messages must be {MAX_MESSAGE_LENGTH:,} characters or fewer."), 400
    try:
        features = vectorizer.transform([message]).toarray().astype(np.float32)
        probability = float(model.predict(features, verbose=0).ravel()[0])
        label = "SPAM" if probability >= 0.5 else "NOT SPAM"
        confidence = probability if label == "SPAM" else 1 - probability
        tokens = re.findall(r"(?u)\b\w\w+\b", message.lower())
        idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
        terms = sorted({t for t in tokens if t in vectorizer.vocabulary_}, key=lambda t: idf.get(t, 0), reverse=True)[:5]
        explanation = ("Terms with strong TF-IDF weighting in this message: " + ", ".join(terms) + "." if terms else "No message terms were recognized by the model vocabulary.")
        preview = message[:180] + ("…" if len(message) > 180 else "")
        with db() as conn:
            conn.execute("INSERT INTO predictions(preview,label,confidence,created_at) VALUES(?,?,?,?)", (preview, label, confidence, datetime.now(timezone.utc).isoformat(timespec="seconds")))
        return jsonify(result=label, confidence=round(confidence * 100, 1), preview=preview, explanation=explanation, status="Model prediction · confidence is not proof that a message is safe")
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Analysis could not be completed. Please try again."), 500

@app.get("/history")
def history():
    return jsonify(items=get_history())

@app.delete("/history")
def clear_history():
    with db() as conn:
        conn.execute("DELETE FROM predictions")
    return jsonify(message="Prediction history cleared.")

@app.get("/model-info")
def model_info():
    if metrics is None:
        return jsonify(ready=False, message="Model artifacts are unavailable. Train with python train.py."), 503
    return jsonify(ready=True, **metrics)

@app.get("/history-page")
def history_page():
    return render_template("history.html")

@app.get("/about")
def about():
    return render_template("about.html", metrics=metrics)

@app.get("/model/training_history.png")
def training_plot():
    from flask import send_file
    from config import PLOT_PATH
    if not PLOT_PATH.is_file():
        return "", 404
    return send_file(PLOT_PATH, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", "5000")))
