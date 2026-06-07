import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model():
    model_name = "./fake_reviews_es"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def test_inference(payload):
    model, tokenizer = load_model()
    data = json.loads(payload)
    inputs = tokenizer(data["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    result = predicted_class
    return result