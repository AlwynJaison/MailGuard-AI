"""Train and evaluate MailGuard's TF-IDF + feed-forward neural network.

CSV datasets need a text and label column. The included starter dataset is
the UCI SMS Spam Collection (SMS, not email). Run with --text-column and
--label-column to use another CSV.
"""
import argparse
import json
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from config import (DATASET_DIR, MAX_FEATURES, MODEL_DIR, MODEL_PATH,
                    PLOT_PATH, RANDOM_STATE, VECTORIZER_PATH, METRICS_PATH)

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def fetch_starter_dataset(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the UCI SMS Spam Collection (SMS dataset)...")
    with urlopen(UCI_URL, timeout=45) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(__import__("io").BytesIO(archive_bytes)) as archive:
        members = [n for n in archive.namelist() if n.endswith("SMSSpamCollection")]
        if not members:
            raise RuntimeError("UCI archive did not contain the expected dataset file.")
        raw = pd.read_csv(archive.open(members[0]), sep="\t", header=None,
                          names=["label", "text"], encoding="utf-8")
    raw.to_csv(destination, index=False)
    return destination


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def load_dataset(path: Path, text_column: str | None, label_column: str | None):
    frame = pd.read_csv(path)
    if text_column is None and {"text", "label"}.issubset(frame.columns):
        text_column, label_column = "text", "label"
    if text_column not in frame.columns or label_column not in frame.columns:
        raise ValueError(f"Columns not found. Available columns: {', '.join(map(str, frame.columns))}")
    frame = frame[[text_column, label_column]].dropna().copy()
    frame[text_column] = frame[text_column].map(clean_text)
    frame = frame[frame[text_column].ne("")].drop_duplicates(subset=[text_column])
    labels = frame[label_column].astype(str).str.strip().str.lower()
    mapping = {"ham": 0, "not spam": 0, "legitimate": 0, "0": 0,
               "spam": 1, "1": 1}
    y = labels.map(mapping)
    if y.isna().any():
        raise ValueError("Labels must be ham/spam, not spam/spam, legitimate/spam, or 0/1.")
    frame = frame.assign(_label=y.astype(int))
    if frame["_label"].nunique() != 2 or frame["_label"].value_counts().min() < 2:
        raise ValueError("The dataset needs at least two examples in each class.")
    return frame[text_column].to_numpy(dtype=str), frame["_label"].to_numpy(dtype=np.int32), len(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="CSV dataset; defaults to UCI starter CSV or fetches it")
    parser.add_argument("--text-column", help="Text column name for a custom CSV")
    parser.add_argument("--label-column", help="Label column name for a custom CSV")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    csv_path = args.dataset or DATASET_DIR / "spam_dataset.csv"
    if not csv_path.exists():
        if args.dataset:
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        csv_path = fetch_starter_dataset(csv_path)
    texts, labels, total = load_dataset(csv_path, args.text_column, args.label_column)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2),
                                 strip_accents="unicode", sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w\w+\b")
    x_train_vec = vectorizer.fit_transform(x_train).toarray().astype("float32")
    x_test_vec = vectorizer.transform(x_test).toarray().astype("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train_vec.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(48, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train_vec, y_train, validation_split=0.15,
                        epochs=args.epochs, batch_size=32, verbose=1,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=3, restore_best_weights=True)])
    probs = model.predict(x_test_vec, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
        "training_samples": int(len(y_train)), "testing_samples": int(len(y_test)),
        "dataset_records": int(total), "dataset_name": "UCI SMS Spam Collection (SMS)" if csv_path.name == "spam_dataset.csv" else csv_path.name,
        "dataset_type": "SMS messages" if csv_path.name == "spam_dataset.csv" else "User-provided CSV",
        "text_column": args.text_column or "text", "label_column": args.label_column or "label",
        "model_type": "TF-IDF (1–2 grams) + feed-forward neural network",
        "threshold": 0.5,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    import pickle
    with VECTORIZER_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(PLOT_PATH, dpi=150); plt.close(fig)
    print("\nHeld-out test metrics")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key.title()}: {metrics[key]:.4f}")
    print(f"Confusion matrix [ham, spam]: {metrics['confusion_matrix']}")
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
