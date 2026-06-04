from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import EmotionPredictor

app = FastAPI(
    title="Emotion Classifier API",
    description="Classifies text into 28 emotions using a fine-tuned DistilBERT model.",
    version="1.0.0"
)

# Load model once at startup
predictor = EmotionPredictor()

class TextInput(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {"text": "I'm so happy for you!"}
        }

class BatchInput(BaseModel):
    texts: list[str]

@app.get("/")
def root():
    return {
        "message": "Emotion Classifier API",
        "endpoints": {
            "/predict": "POST - classify a single text",
            "/predict/batch": "POST - classify multiple texts",
            "/health": "GET - check API status"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "distilbert-finetuned-goemotions"}

@app.post("/predict")
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return predictor.predict(input.text)

@app.post("/predict/batch")
def predict_batch(input: BatchInput):
    if len(input.texts) > 50:
        raise HTTPException(status_code=400, detail="Max 50 texts per batch")
    return [predictor.predict(text) for text in input.texts]