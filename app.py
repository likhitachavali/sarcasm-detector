"""
app.py — Flask API for Sarcasm Detection (DistilBERT)
Run AFTER training: python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os

app = Flask(__name__, static_folder="frontend")
CORS(app)

MODEL_DIR = "./model"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
    raise FileNotFoundError("No trained model found in ./model/\nPlease run python train.py first!")

print("Loading fine-tuned DistilBERT model...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()
print("Model ready ✅")


def predict(text: str):
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).squeeze().cpu().tolist()

    label      = "Sarcastic" if probs[1] > probs[0] else "Not Sarcastic"
    confidence = max(probs) * 100
    return {
        "label":           label,
        "confidence":      round(confidence, 2),
        "prob_sarcastic":  round(probs[1] * 100, 2),
        "prob_not":        round(probs[0] * 100, 2),
    }


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > 512:
        return jsonify({"error": "Text too long — keep it under 512 characters."}), 400
    return jsonify(predict(text))


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
