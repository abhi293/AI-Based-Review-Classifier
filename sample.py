import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Example classifier

# Sample dataset
train_texts = ["This product is great", "I don't like this item", "Amazing quality!", "Worst experience ever"]
train_labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Create and fit the vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)

# Train a classification model (Logistic Regression in this case)
model = LogisticRegression()
model.fit(X_train, train_labels)

# Save both vectorizer and model
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")

print("Model and vectorizer saved successfully!")
