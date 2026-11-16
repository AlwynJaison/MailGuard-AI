# MailGuard AI — Intelligent Email Spam Detection Using Neural Networks

A small Flask application for classifying pasted message text as **SPAM** or **NOT SPAM**. It uses a trained feed-forward neural network over TF-IDF features; predictions are not hardcoded. Model training runs separately from the web server.

> **Dataset note:** the included first-run starter download is the UCI SMS Spam Collection. It contains SMS messages, not emails. Results on this dataset do not establish performance on real email.

## Requirements and setup

Python 3.10–3.12 is recommended. From this directory:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The public UCI SMS Spam Collection CSV and a trained model are included in this project folder. If the CSV is removed, `train.py` can download it again. To retrain, then start the local server:

```bash
python train.py
python app.py
```

Open <http://127.0.0.1:5000>. The trained network and preprocessing vectorizer are stored under `model/`; the server loads them at startup and never trains on startup. Training prints the held-out metrics and creates a training plot.

### Train from another labelled CSV

The CSV needs a text column and labels representing ham/spam (also accepts `not spam`/`spam`, `legitimate`/`spam`, or `0`/`1`). By default, columns named `text` and `label` are used. Choose other columns explicitly:

```bash
python train.py --dataset dataset/my_messages.csv --text-column body --label-column category --epochs 12
```

The trainer drops missing values, normalizes whitespace, removes duplicate message text, maps labels, makes a stratified 80/20 train/test split, and fits TF-IDF on training data only. The ANN uses two dense ReLU layers, dropout, a sigmoid output, Adam, binary cross-entropy, and early stopping against a validation split.

## What is included

- `app.py` — Flask pages and JSON endpoints (`POST /predict`, `GET /history`, `DELETE /history`, `GET /model-info`).
- `train.py` — dataset loading, neural network training and held-out evaluation.
- `templates/` and `static/` — responsive dashboard, analysis results, history, and model information.
- `database/history.db` — created automatically for local prediction history; it is not included in the repository.
- `model/` — the included trained artifacts (`spam_classifier.keras`, `vectorizer.pkl`, `evaluation.json`, and `training_history.png`). Running `train.py` regenerates these files.

## Evaluation and interpretation

The included trained model was evaluated on a stratified held-out split of the UCI SMS Spam Collection after cleanup (5,158 records: 4,126 train and 1,032 test):

| Metric | Held-out result |
| --- | ---: |
| Accuracy | 0.9806 |
| Precision | 0.9576 |
| Recall | 0.8828 |
| F1 | 0.9187 |
| Confusion matrix (true ham/spam × predicted ham/spam) | `[[899, 5], [15, 113]]` |

These are dataset results, not an estimate of real-world email performance. The model information screen reports metrics stored by the trainer. The explanation lists recognized message terms with high inverse-document-frequency weighting. This is a feature summary, not a causal explanation of the neural network output. Confidence is not proof that a message is safe.

## Example inputs

Spam-style example:

> Congratulations! You have won a free cash prize. Click here to claim your reward now!

Legitimate-style example:

> Hello, please find the meeting notes attached. Let me know if you have any questions.

The classifier is a learning baseline and can be wrong, especially on short, unusual, or email text different from the SMS training data. Do not use its result as a security guarantee.

## API checks

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"message":"Congratulations, claim your prize now!"}'
curl http://127.0.0.1:5000/history
curl http://127.0.0.1:5000/model-info
curl -X DELETE http://127.0.0.1:5000/history
```

An empty message returns HTTP 400. When model files are absent, the home page still loads and `/predict` returns HTTP 503 with setup guidance.
