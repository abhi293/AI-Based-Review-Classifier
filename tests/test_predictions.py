import json
import requests

URL = "http://127.0.0.1:5000/predict"

def test_prediction(text):
    response = requests.post(URL, json={"text": text})
    return response.json()

if __name__ == "__main__":
    sample_texts = [
        "The product was amazing! I loved it.",
        "This is the worst experience I've ever had.",
        "It's okay, not great but not bad either."
    ]

    for text in sample_texts:
        print(f"Input: {text}")
        print("Prediction:", test_prediction(text))
