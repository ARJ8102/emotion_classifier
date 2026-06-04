# 🎭 Emotion Classifier

A fine-tuned DistilBERT model that classifies text into 28 emotion categories, trained on Google's [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset (58K Reddit comments).

<!-- Add your screenshot here -->
[Demo](assets/demo.png)

## Results

| Model | Accuracy |
|---|---|
| TF-IDF + Logistic Regression (Baseline) | 41.95% |
| **Fine-tuned DistilBERT** | **58.12%** |

The fine-tuned transformer achieves a **38% relative improvement** over the baseline, with strong performance on common emotions (gratitude 82% F1, amusement 80%, admiration 71%) and expected difficulty on rare categories (grief, nervousness, pride).

### Key Findings

- **Neutral bias**: The model tends to over-predict "neutral" for subtle or context-dependent emotions
- **Close emotion confusion**: Related emotions bleed into each other (anger ↔ annoyance, admiration ↔ approval)
- **Sarcasm and irony**: Remain challenging — the model takes text at face value
- Some human labels in GoEmotions are debatable, suggesting the model's actual performance may be slightly better than reported

## Features

- **28 emotion categories**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral
- **REST API**: FastAPI endpoint with batch support and Swagger docs
- **Web UI**: Streamlit interface with confidence scores and top-3 predictions
- **Docker support**: Containerized for deployment
- **Evaluation suite**: Per-class F1 comparison, confusion matrix, error analysis

## Project Structure

```
emotion-classifier/
├── src/
│   ├── baseline.py       # TF-IDF + Logistic Regression baseline
│   ├── train.py          # DistilBERT fine-tuning pipeline
│   ├── predict.py        # Inference module
│   └── evaluate.py       # Model comparison & visualization
├── models/               # Saved model weights & results
├── notebooks/            # EDA and exploration
├── app.py                # FastAPI REST API
├── ui.py                 # Streamlit web interface
├── Dockerfile            # Container config
├── requirements.txt      # Dependencies
└── README.md
```

## Quick Start

### Setup

```bash
git clone https://github.com/ARJ8102/emotion-classifier.git
cd emotion-classifier
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Web UI

```bash
streamlit run ui.py
```

Opens at `http://localhost:8501`

### Run the API

```bash
uvicorn app:app --reload
```

API docs at `http://localhost:8000/docs`

#### Example API call

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy for you!"}'
```

```json
{
  "text": "I am so happy for you!",
  "predicted_emotion": "joy",
  "confidence": 0.8234,
  "top_3": [
    {"emotion": "joy", "confidence": 0.8234},
    {"emotion": "admiration", "confidence": 0.0891},
    {"emotion": "excitement", "confidence": 0.0312}
  ]
}
```

### Run with Docker

```bash
docker build -t emotion-classifier .
docker run -p 8000:8000 emotion-classifier
```

## Approach

1. **Baseline**: TF-IDF vectorization (10K features, unigrams + bigrams) with Logistic Regression and balanced class weights to handle label imbalance
2. **Fine-tuning**: DistilBERT with a classification head, trained for 3 epochs with AdamW optimizer, linear warmup schedule, and gradient clipping
3. **Evaluation**: Per-class metrics, confusion matrix on top-10 emotions, and qualitative error analysis on misclassified samples

### Training Config

| Parameter | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Max sequence length | 64 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Warmup ratio | 10% |
| Optimizer | AdamW |

## Tech Stack

- **ML**: PyTorch, HuggingFace Transformers, scikit-learn
- **Data**: HuggingFace Datasets (GoEmotions)
- **API**: FastAPI, Uvicorn
- **UI**: Streamlit
- **Deployment**: Docker

## Future Improvements

- Multi-label classification (a comment can express multiple emotions simultaneously)
- Data augmentation for rare emotion categories
- Experiment with larger models (RoBERTa, DeBERTa)
- Add Weights & Biases experiment tracking
- Deploy on Streamlit Cloud / HuggingFace Spaces

## License

MIT
