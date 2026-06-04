import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import json

# ---------- CONFIG ----------
# Keep all hyperparameters in one place (clean code practice)
CONFIG = {
    "model_name": "distilbert-base-uncased",
    "num_labels": 28,
    "max_length": 64,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"Using device: {CONFIG['device']}")

def load_and_prepare_data(tokenizer):
    """Load GoEmotions, simplify to single-label, and tokenize."""
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    emotion_labels = dataset['train'].features['labels'].feature.names

    def simplify(example):
        example['label'] = example['labels'][0]
        return example

    def tokenize(example):
        tokens = tokenizer(
            example['text'],
            padding='max_length',
            truncation=True,
            max_length=CONFIG['max_length']
        )
        tokens['label'] = example['label']
        return tokens

    # Process each split
    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].map(simplify)
        dataset[split] = dataset[split].map(tokenize)
        # Keep only the columns PyTorch needs
        dataset[split].set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'label']
        )

    return dataset, emotion_labels


def train_model(model, train_loader, val_loader, emotion_labels):
    """Fine-tune DistilBERT with proper training loop."""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate']
    )

    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_accuracy = 0
    history = {"train_loss": [], "val_accuracy": []}

    for epoch in range(CONFIG['epochs']):
        # --- Training phase ---
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Validation phase ---
        val_accuracy, val_report = evaluate_model(model, val_loader, emotion_labels)
        history['val_accuracy'].append(val_accuracy)

        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'models/best_distilbert.pt')
            print(f"  New best model saved! ({best_accuracy:.4f})")

    return history, best_accuracy

def evaluate_model(model, data_loader, emotion_labels):
    """Evaluate model and return accuracy + report."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=emotion_labels,
        zero_division=0
    )

    return accuracy, report




if __name__ == "__main__":
    # 1. Load tokenizer and data
    print("Loading tokenizer and data...")
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['model_name'])
    dataset, emotion_labels = load_and_prepare_data(tokenizer)

    # 2. Create data loaders
    train_loader = DataLoader(dataset['train'], batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset['validation'], batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(dataset['test'], batch_size=CONFIG['batch_size'])

    # 3. Load model
    print("Loading DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=CONFIG['num_labels']
    )
    model.to(CONFIG['device'])

    # 4. Train
    print(f"\nStarting training for {CONFIG['epochs']} epochs...")
    history, best_accuracy = train_model(model, train_loader, val_loader, emotion_labels)

    # 5. Final evaluation on test set
    print("\n--- Final Test Set Evaluation ---")
    model.load_state_dict(torch.load('models/best_distilbert.pt'))
    test_accuracy, test_report = evaluate_model(model, test_loader, emotion_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"\n{test_report}")

    # 6. Save results for comparison later
    results = {
        "config": CONFIG,
        "best_val_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
        "baseline_accuracy": 0.4195,
        "history": history
    }
    # device is not JSON serializable, convert it
    results['config']['device'] = str(results['config']['device'])

    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 7. Quick sanity check
    print("\n--- Sample Predictions ---")
    sample_texts = [
        "I'm so happy for you!",
        "This makes me angry",
        "I'm really scared about tomorrow",
        "Thank you so much, you're amazing"
    ]

    model.eval()
    for text in sample_texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(CONFIG['device']) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        pred = torch.argmax(output.logits, dim=-1).item()
        print(f"  \"{text}\" → {emotion_labels[pred]}")

    print(f"\n Baseline: {0.4195:.4f} → DistilBERT: {test_accuracy:.4f}")