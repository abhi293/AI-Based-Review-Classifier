import pandas as pd
import joblib
from training.preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression
import os

# Load training data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(BASE_DIR, '../data/train.csv')

X, y, vectorizer = preprocess_data(train_file)  # Now getting vectorizer too

# Train model
def train_logistic_model(X, y, vectorizer):  # Accept vectorizer as argument
    model = LogisticRegression()
    model.fit(X, y)

    # Save model & vectorizer
    joblib.dump(model, os.path.join(BASE_DIR, '../model.pkl'))
    joblib.dump(vectorizer, os.path.join(BASE_DIR, '../vectorizer.pkl'))

    print("Model training complete. Saved as model.pkl and vectorizer.pkl")

# If running this script directly, train the model
if __name__ == "__main__":
    train_logistic_model(X, y, vectorizer)
