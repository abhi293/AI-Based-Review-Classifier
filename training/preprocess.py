import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(filepath):
    df = pd.read_csv(filepath, header=None, names=['label', 'title', 'text'])

    # Handle missing values in the text column
    df['text'] = df['text'].fillna("")

    # Convert text to feature vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])  # Use 'text' column, not 'title'
    y = df['label']

    print(f"Shape of X: {X.shape}")  # Debugging
    print(f"Shape of y: {y.shape}")  # Debugging

    return X, y, vectorizer  # Return vectorizer along with X, y
