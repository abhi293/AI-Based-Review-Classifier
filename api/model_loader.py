import joblib
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '../model.pkl')
    return joblib.load(model_path)

def load_vectorizer():
    vectorizer_path = os.path.join(os.path.dirname(__file__), '../vectorizer.pkl')
    return joblib.load(vectorizer_path)
