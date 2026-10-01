# 🧠 SarcasmSense — AI Sarcasm Detector

Detects sarcasm in text using a **fine-tuned BERT** model trained on the
News Headlines Sarcasm Dataset (~28,000 labeled headlines).

---

## 📁 Project Structure

```
sarcasm-detector/
├── train.py          ← Fine-tune BERT (run this first!)
├── app.py            ← Flask backend API
├── requirements.txt  ← Python dependencies
├── data/             ← Dataset auto-downloaded here
├── model/            ← Fine-tuned model saved here
└── frontend/
    └── index.html    ← Web UI
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
- Downloads the dataset automatically (~28k headlines)
- Fine-tunes BERT for 3 epochs (~30–60 min on CPU, ~5 min on GPU)
- Saves best model to `./model/`
- Expected accuracy: **~93–95%**

### 3. Start the Flask server
```bash
python app.py
```

### 4. Open the app
Visit → [http://localhost:5000](http://localhost:5000)

---

## 🔬 How It Works

| Step | Detail |
|------|--------|
| **Model** | `bert-base-uncased` fine-tuned for binary classification |
| **Dataset** | News Headlines Sarcasm Dataset (Kaggle / GitHub) |
| **Classes** | `Sarcastic` vs `Not Sarcastic` |
| **Input** | Raw text (max 512 chars) |
| **Output** | Label + confidence probabilities |

---

## 📡 API

**POST** `/analyze`
```json
// Request
{ "text": "Oh great, another Monday. Just what I needed." }

// Response
{
  "label":          "Sarcastic",
  "confidence":     94.2,
  "prob_sarcastic": 94.2,
  "prob_not":       5.8
}
```

---

## 💡 Tips
- Use **GPU** if available — training is ~10x faster
- The model works best on English text
- Press **Ctrl+Enter** in the text box to analyze quickly
