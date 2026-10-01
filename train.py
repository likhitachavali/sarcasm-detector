"""
train.py — Fine-tune DistilBERT for Sarcasm Detection
Mixed dataset: News Headlines + Conversational sarcasm examples
Estimated time: ~30-40 min on CPU | ~5 min on GPU
Run: python train.py
"""

# LOAD DATASET
# 28k News Headlines + Custom Conversational Examples
#       ↓
# SPLIT DATA
# 85% Training | 15% Validation
#       ↓
# TOKENIZE ALL TEXT
# Convert every sentence to token IDs
#       ↓
# FINE-TUNE DISTILBERT
# For each batch:
#   → Forward pass through model
#   → Calculate loss (how wrong the prediction was)
#   → Backpropagation (adjust weights to reduce loss)
#   → Repeat for all batches = 1 Epoch
#       ↓
# EVALUATE ON VALIDATION SET
# Check accuracy on unseen data
# Save model if accuracy improved
#       ↓
# REPEAT FOR EPOCH 2
#       ↓
# SAVE BEST MODEL to ./model/

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 2
LR           = 2e-5
SAVE_DIR     = "./model"
DATA_PATH    = "./data/sarcasm.json"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. DOWNLOAD DATASET
# ─────────────────────────────────────────────
def download_dataset():
    url = (
        "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection"
        "/master/Sarcasm_Headlines_Dataset.json"
    )
    if os.path.exists(DATA_PATH):
        print("Dataset already exists, skipping download.")
        return
    print("Downloading dataset...")
    response = requests.get(url)
    response.raise_for_status()
    lines   = response.text.strip().split("\n")
    records = [json.loads(line) for line in lines]
    with open(DATA_PATH, "w") as f:
        json.dump(records, f)
    print(f"Dataset saved to {DATA_PATH} ({len(records)} records)")


def load_dataset():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    headlines = [d["headline"] for d in data]
    labels    = [d["is_sarcastic"] for d in data]
    return headlines, labels


# ─────────────────────────────────────────────
# 2. CONVERSATIONAL SARCASM EXAMPLES
# ─────────────────────────────────────────────
def get_conversational_data():
    sarcastic = [
        "oh great, another monday. just what i needed.",
        "sure, i totally love working overtime every weekend.",
        "yeah because getting stuck in traffic for 2 hours is my favourite hobby.",
        "oh wow, what a surprise, it's raining again.",
        "yeah i absolutely love it when my alarm goes off at 6am.",
        "oh fantastic, the wifi is down again. best day ever.",
        "sure, because i have nothing better to do than wait in line for an hour.",
        "oh brilliant, my phone died right when i needed it most.",
        "yeah nothing i love more than a monday morning meeting.",
        "oh great, the coffee machine is broken. this day just keeps getting better.",
        "sure i'd love to sit in traffic for three hours, sounds amazing.",
        "oh wow thanks for the two days notice, super helpful.",
        "yeah because sleeping 4 hours is totally enough for anyone.",
        "oh perfect, another software update right when i'm in the middle of work.",
        "yeah i just love when people show up late to everything.",
        "oh fantastic, they cancelled the show right at the season finale.",
        "sure, because paying full price for a broken product is totally fine.",
        "oh great another password reset, this is exactly what i wanted to do today.",
        "yeah nothing better than a group project where only one person does all the work.",
        "oh wow they added more ads, what a great improvement.",
        "sure i love when my order arrives three weeks late.",
        "oh brilliant idea to schedule the meeting at 8am on a friday.",
        "yeah because of course the atm is out of cash when i need it.",
        "oh wonderful, more homework on a holiday weekend.",
        "sure, who needs sleep anyway.",
        "oh great the bus left exactly one minute before i arrived.",
        "yeah i totally enjoy being put on hold for 45 minutes.",
        "oh wow another fee i didn't know about, what a pleasant surprise.",
        "sure because nothing says fun like doing taxes on a sunday.",
        "oh brilliant, they raised the prices and made the portions smaller.",
        "yeah because my opinion totally matters in this meeting.",
        "oh sure, i love when plans get cancelled last minute.",
        "wow what a great time to have no umbrella.",
        "yeah i just love surprise deadlines, they're my favourite.",
        "oh amazing, the printer is out of ink again.",
        "sure, i definitely wanted to redo all my work from scratch.",
        "oh wow my favourite, a three hour lecture on a friday afternoon.",
        "yeah nothing like cold food delivered an hour late.",
        "oh great, they changed the app layout again for no reason.",
        "sure because obviously i have unlimited free time.",
    ]

    not_sarcastic = [
        "the new library opened downtown last week.",
        "she passed her driving test on the first attempt.",
        "the weather forecast says it will be sunny tomorrow.",
        "the team finished the project ahead of schedule.",
        "he got a promotion at work this month.",
        "the new restaurant on fifth street has great reviews.",
        "scientists have made a breakthrough in cancer research.",
        "the flight landed safely despite the stormy weather.",
        "she scored top marks in her final exams.",
        "the city announced free public transport on weekends.",
        "the new software update fixed all the previous bugs.",
        "they launched a new recycling program in the neighbourhood.",
        "the match ended in a draw after extra time.",
        "the hospital received a large donation for new equipment.",
        "the annual festival attracted thousands of visitors this year.",
        "she learned to play the guitar in just three months.",
        "the new park opened with facilities for all age groups.",
        "the volunteers cleaned up the entire beach in one afternoon.",
        "the company announced record profits this quarter.",
        "the students organised a successful charity drive.",
        "the bridge construction was completed two months early.",
        "the new bus route connects the two largest neighbourhoods.",
        "the local team won the regional championship.",
        "the museum extended its opening hours for the summer.",
        "the power outage was restored within two hours.",
        "she received a scholarship to study abroad next year.",
        "the new app makes it easy to track daily expenses.",
        "the community garden produced enough vegetables for the whole street.",
        "the conference brought together experts from around the world.",
        "the rescue team found all missing hikers safe and unharmed.",
        "the school introduced a new coding programme for students.",
        "the concert was sold out within minutes of tickets going live.",
        "the river cleanup project removed tons of waste this year.",
        "the new train line cut commute times in half.",
        "she donated a portion of her salary to the local shelter.",
        "the research team published their findings in a top journal.",
        "the city planted thousands of new trees this spring.",
        "the tournament was won by the youngest player in history.",
        "the app update improved battery life significantly.",
        "the new policy gives employees more flexible working hours.",
    ]

    texts  = sarcastic + not_sarcastic
    labels = [1] * len(sarcastic) + [0] * len(not_sarcastic)
    return texts, labels


# ─────────────────────────────────────────────
# 3. PYTORCH DATASET
# ─────────────────────────────────────────────
class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        logits  = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def eval_epoch(model, loader):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds), all_preds, all_labels


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
def main():
    download_dataset()
    texts, labels = load_dataset()
    print(f"Total headline samples: {len(texts)}")

    # Use 40% of headlines
    texts, _, labels, _ = train_test_split(
        texts, labels, test_size=0.70, random_state=42, stratify=labels
    )

    # Mix in conversational examples (repeated 5x to give them more weight)
    conv_texts, conv_labels = get_conversational_data()
    texts  = texts  + conv_texts * 3
    labels = labels + conv_labels * 3
    print(f"Mixed dataset size: {len(texts)} samples (headlines + conversational)")

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    train_dl = DataLoader(SarcasmDataset(X_train, y_train, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(SarcasmDataset(X_val,   y_val,   tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_val_acc = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}\nEPOCH {epoch}/{EPOCHS}\n{'='*50}")

        train_loss, train_acc = train_epoch(model, train_dl, optimizer, scheduler)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_dl)

        print(f"Train  → Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        print(f"Val    → Loss: {val_loss:.4f}   | Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"✅ Best model saved! (Val Acc: {val_acc*100:.2f}%)")

    print(f"\n{'='*50}")
    print(f"Training complete! Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {SAVE_DIR}/")
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=["Not Sarcastic", "Sarcastic"]))


if __name__ == "__main__":
    main()
