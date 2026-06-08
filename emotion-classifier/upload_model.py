import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=28
)
model.load_state_dict(torch.load("models/best_distilbert.pt", map_location="cpu"))

model.save_pretrained("models/hf_model")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained("models/hf_model")

print("Done! Now upload with the CLI command.")