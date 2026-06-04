from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import pickle
import os

def load_and_prepare_data():
    """Load GoEmotions and simplify to single-label."""
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    
    emotion_labels = dataset['train'].features['labels'].feature.names
    
    def simplify(example):
        # Take first label only (simplification)
        example['label'] = example['labels'][0]
        return example
    
    train = dataset['train'].map(simplify)
    val = dataset['validation'].map(simplify)
    test = dataset['test'].map(simplify)
    
    return train, val, test, emotion_labels




def train_baseline(train_data, val_data, emotion_labels):
    """Train TF-IDF + Logistic Regression baseline."""
    
    # Step 1: Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_data['text'])
    X_val = vectorizer.transform(val_data['text'])
    
    y_train = train_data['label']
    y_val = val_data['label']
    
    # Step 2: Train Logistic Regression
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Step 3: Evaluate on validation set
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(
        y_val, y_pred,
        target_names=emotion_labels,
        zero_division=0
    )
    
    print(f"\nBaseline Validation Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    
    return model, vectorizer, accuracy





def save_baseline(model, vectorizer):
    """Save model and vectorizer for later comparison."""
    os.makedirs('models', exist_ok=True)
    with open('models/baseline_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Baseline model saved.")

if __name__ == "__main__":
    print("Loading data...")
    train, val, test, emotion_labels = load_and_prepare_data()
    
    print("Training baseline...")
    model, vectorizer, accuracy = train_baseline(train, val, emotion_labels)
    
    save_baseline(model, vectorizer)
    
    # Quick sanity check - try a few predictions
    sample_texts = [
        "I'm so happy for you!",
        "This makes me angry",
        "I'm really scared about tomorrow",
        "Thank you so much, you're amazing"
    ]
    
    X_sample = vectorizer.transform(sample_texts)
    predictions = model.predict(X_sample)
    
    print("\n--- Sample Predictions ---")
    for text, pred in zip(sample_texts, predictions):
        print(f"  \"{text}\" → {emotion_labels[pred]}")