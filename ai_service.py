from api.model_loader import load_model
from api.app import predict

# Initialize model
model = load_model()

def classify_review(text):
    return predict(model, text)
