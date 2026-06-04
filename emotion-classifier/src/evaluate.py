import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset

# ---- Load everything ----
def load_all():
    """Load both models, data, and labels."""
    
    # Config (must match training)
    model_name = "distilbert-base-uncased"
    max_length = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    emotion_labels = dataset['train'].features['labels'].feature.names
    
    def simplify(example):
        example['label'] = example['labels'][0]
        return example
    
    test_data = dataset['test'].map(simplify)
    
    # Baseline model
    with open('models/baseline_model.pkl', 'rb') as f:
        baseline_model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Transformer model
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    transformer = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=28
    )
    transformer.load_state_dict(torch.load('models/best_distilbert.pt', map_location=device))
    transformer.to(device)
    transformer.eval()
    
    return (baseline_model, vectorizer, transformer, tokenizer,
            test_data, emotion_labels, device, max_length)


# ---- Get predictions from both models ----
def get_predictions(baseline_model, vectorizer, transformer, tokenizer,
                    test_data, device, max_length):
    
    texts = test_data['text']
    true_labels = test_data['label']
    
    # Baseline predictions
    X_test = vectorizer.transform(texts)
    baseline_preds = baseline_model.predict(X_test)
    
    # Transformer predictions
    transformer_preds = []
    batch_size = 32
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors='pt',
            padding=True, truncation=True, max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = transformer(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        transformer_preds.extend(preds)
    
    return true_labels, baseline_preds, np.array(transformer_preds)


# ---- Visualization 1: Accuracy comparison bar chart ----
def plot_accuracy_comparison(true_labels, baseline_preds, transformer_preds,
                            emotion_labels):
    
    baseline_acc = np.mean(np.array(baseline_preds) == np.array(true_labels))
    transformer_acc = np.mean(np.array(transformer_preds) == np.array(true_labels))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ['TF-IDF + LogReg\n(Baseline)', 'Fine-tuned\nDistilBERT']
    accs = [baseline_acc, transformer_acc]
    colors = ['#ff6b6b', '#51cf66']
    
    bars = ax.bar(models, accs, color=colors, width=0.5, edgecolor='black')
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', fontsize=14, fontweight='bold')
    
    ax.set_ylim(0, 0.75)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Comparison: Emotion Classification', fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('models/accuracy_comparison.png', dpi=150)
    plt.show()
    print(f"Baseline: {baseline_acc:.4f} | DistilBERT: {transformer_acc:.4f}")


# ---- Visualization 2: Per-class F1 comparison ----
def plot_per_class_f1(true_labels, baseline_preds, transformer_preds,
                      emotion_labels):
    
    baseline_report = classification_report(
        true_labels, baseline_preds, target_names=emotion_labels,
        output_dict=True, zero_division=0
    )
    transformer_report = classification_report(
        true_labels, transformer_preds, target_names=emotion_labels,
        output_dict=True, zero_division=0
    )
    
    # Get F1 scores for each emotion
    emotions = emotion_labels
    baseline_f1 = [baseline_report[e]['f1-score'] for e in emotions]
    transformer_f1 = [transformer_report[e]['f1-score'] for e in emotions]
    
    # Sort by transformer F1 for readability
    sorted_idx = np.argsort(transformer_f1)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(emotions))
    height = 0.35
    
    ax.barh(y - height/2, [baseline_f1[i] for i in sorted_idx],
            height, label='Baseline', color='#ff6b6b', alpha=0.8)
    ax.barh(y + height/2, [transformer_f1[i] for i in sorted_idx],
            height, label='DistilBERT', color='#51cf66', alpha=0.8)
    
    ax.set_yticks(y)
    ax.set_yticklabels([emotions[i] for i in sorted_idx])
    ax.set_xlabel('F1 Score')
    ax.set_title('Per-Emotion F1 Score Comparison')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('models/per_class_f1.png', dpi=150)
    plt.show()


# ---- Visualization 3: Confusion matrix for transformer ----
def plot_confusion_matrix(true_labels, transformer_preds, emotion_labels):
    
    # Only show top 10 most common emotions (full 28x28 is unreadable)
    from collections import Counter
    label_counts = Counter(true_labels)
    top_10 = [label for label, _ in label_counts.most_common(10)]
    top_names = [emotion_labels[i] for i in top_10]
    
    # Filter to only samples with top-10 labels
    mask = np.isin(true_labels, top_10)
    filtered_true = np.array(true_labels)[mask]
    filtered_pred = np.array(transformer_preds)[mask]
    
    cm = confusion_matrix(filtered_true, filtered_pred, labels=top_10)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=top_names, yticklabels=top_names, ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix (Top 10 Emotions, Normalized)')
    
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150)
    plt.show()


# ---- Error analysis ----
def error_analysis(true_labels, transformer_preds, test_data, emotion_labels):
    """Find interesting mistakes the model makes."""
    
    texts = test_data['text']
    errors = []
    
    for i in range(len(true_labels)):
        if true_labels[i] != transformer_preds[i]:
            errors.append({
                'text': texts[i],
                'true': emotion_labels[true_labels[i]],
                'predicted': emotion_labels[transformer_preds[i]]
            })
    
    print(f"\nTotal errors: {len(errors)} / {len(true_labels)} ({len(errors)/len(true_labels):.1%})")
    print("\n--- 15 Interesting Misclassifications ---")
    
    # Show a diverse sample
    np.random.seed(42)
    sample = np.random.choice(len(errors), min(15, len(errors)), replace=False)
    
    for idx in sample:
        e = errors[idx]
        print(f"\n  Text: \"{e['text'][:100]}\"")
        print(f"  True: {e['true']} | Predicted: {e['predicted']}")
    
    # Save errors for the notebook
    import pandas as pd
    error_df = pd.DataFrame(errors)
    error_df.to_csv('models/error_analysis.csv', index=False)
    print(f"\nFull error analysis saved to models/error_analysis.csv")


# ---- Main ----
if __name__ == "__main__":
    print("Loading models and data...")
    (baseline_model, vectorizer, transformer, tokenizer,
     test_data, emotion_labels, device, max_length) = load_all()
    
    print("Getting predictions...")
    true_labels, baseline_preds, transformer_preds = get_predictions(
        baseline_model, vectorizer, transformer, tokenizer,
        test_data, device, max_length
    )
    
    print("\n1. Accuracy Comparison")
    plot_accuracy_comparison(true_labels, baseline_preds, transformer_preds,
                           emotion_labels)
    
    print("\n2. Per-Class F1 Comparison")
    plot_per_class_f1(true_labels, baseline_preds, transformer_preds,
                     emotion_labels)
    
    print("\n3. Confusion Matrix")
    plot_confusion_matrix(true_labels, transformer_preds, emotion_labels)
    
    print("\n4. Error Analysis")
    error_analysis(true_labels, transformer_preds, test_data, emotion_labels)
    
    print("\nAll visualizations saved to models/ folder.")