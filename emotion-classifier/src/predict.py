import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class EmotionPredictor:
    """Loads the fine-tuned model and runs inference."""
    
    def __init__(self, model_path="models/best_distilbert.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "distilbert-base-uncased"
        self.max_length = 64
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral"
        ]
        
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=28
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str) -> dict:
        """Predict emotion for a single text."""
        inputs = self.tokenizer(
            text, return_tensors="pt",
            padding=True, truncation=True, max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        top_idx = torch.argmax(probs).item()
        
        # Get top 3 predictions
        top3_values, top3_indices = torch.topk(probs, 3)
        top3 = [
            {"emotion": self.emotion_labels[i], "confidence": round(v.item(), 4)}
            for v, i in zip(top3_values, top3_indices)
        ]
        
        return {
            "text": text,
            "predicted_emotion": self.emotion_labels[top_idx],
            "confidence": round(probs[top_idx].item(), 4),
            "top_3": top3
        }