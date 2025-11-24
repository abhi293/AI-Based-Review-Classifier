# 🎓 Model Training Guide

This guide will walk you through the complete process of training the AI-Based Review Classifier model from scratch.

---

## 📋 Prerequisites

### 1. System Requirements
- **Python 3.8+** installed
- **pip** package manager
- At least **2GB RAM** (4GB+ recommended)
- **500MB disk space** for dependencies

### 2. Data Requirements
Ensure you have the training dataset in the correct location:
- File: `data/train.csv`
- Format: CSV with columns `[label, title, text]`
  - `label`: 1 (negative) or 2 (positive)
  - `title`: Review title (optional)
  - `text`: Review text content

---

## ⚙️ Setup Instructions

### Step 1: Install Dependencies

Navigate to the project root directory and install all required packages:

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `scikit-learn` - Machine learning library (Logistic Regression, TF-IDF)
- `pandas` - Data manipulation and CSV handling
- `numpy` - Numerical computations
- `joblib` - Model serialization
- `Flask` - Web API framework
- `fastapi` & `uvicorn` - Alternative API framework (optional)
- `requests` - HTTP testing library

---

## 🚀 Training the Model

### Step 2: Verify Your Data

Check if your training data is balanced:

```bash
python check_data_balance.py
```

This will show you the distribution of positive vs. negative reviews. An imbalanced dataset may affect model performance.

### Step 3: Train the Model

Run the training script from the project root:

```bash
python -m training.train_model
```

**What happens during training:**
1. ✅ Loads `data/train.csv`
2. ✅ Preprocesses text data (handles missing values)
3. ✅ Converts text to TF-IDF feature vectors
4. ✅ Trains Logistic Regression model
5. ✅ Saves two files:
   - `model.pkl` - Trained Logistic Regression model
   - `vectorizer.pkl` - TF-IDF vectorizer with learned vocabulary

**Expected Output:**
```
Shape of X: (N, M)  # N = number of samples, M = vocabulary size
Shape of y: (N,)
Model training complete. Saved as model.pkl and vectorizer.pkl
```

### Step 4: Evaluate the Model

Test your model's performance on the test dataset:

```bash
python -m training.evaluate
```

This will output:
- **Accuracy score** (percentage of correct predictions)
- Model performance metrics

---

## 🧪 Testing Your Trained Model

### Option 1: Quick Test with Sample Script

```bash
python sample.py
```

This runs a quick test with sample data.

### Option 2: API Testing

1. **Start the Flask API:**
   ```bash
   cd api
   python app.py
   ```

2. **Test with a POST request:**
   ```bash
   curl -X POST http://localhost:5000/predict ^
        -H "Content-Type: application/json" ^
        -d "{\"text\": \"This product is amazing!\"}"
   ```

   **Expected Response:**
   ```json
   {
     "prediction": "positive",
     "confidence": "98.00%"
   }
   ```

### Option 3: Run Unit Tests

```bash
python tests/test_predictions.py
python tests/test_preprocessing.py
```

---

## 📊 Understanding the Training Process

### Algorithm: Logistic Regression
- **Type**: Linear classification algorithm
- **Purpose**: Binary sentiment classification
- **Advantages**:
  - Fast training and prediction
  - Interpretable coefficients
  - Works well with high-dimensional sparse data (TF-IDF)
  - Low memory footprint

### Feature Engineering: TF-IDF
- **TF (Term Frequency)**: How often a word appears in a review
- **IDF (Inverse Document Frequency)**: How unique/rare a word is across all reviews
- **Result**: Sparse numerical vectors representing text

**Example:**
```
Text: "This product is great!"
→ TF-IDF Vector: [0.0, 0.52, 0.0, 0.85, 0.0, ...]
```

---

## 🔧 Customizing Training Parameters

### Modify TF-IDF Settings

Edit `training/preprocess.py`:

```python
vectorizer = TfidfVectorizer(
    max_features=5000,      # Limit vocabulary size
    min_df=2,               # Ignore rare words (appear in < 2 docs)
    max_df=0.8,             # Ignore common words (appear in > 80% docs)
    ngram_range=(1, 2)      # Use unigrams and bigrams
)
```

### Modify Model Settings

Edit `training/train_model.py`:

```python
model = LogisticRegression(
    max_iter=1000,          # Maximum iterations
    C=1.0,                  # Regularization strength (lower = stronger)
    solver='lbfgs',         # Optimization algorithm
    random_state=42         # For reproducibility
)
```

---

## 📦 Model Artifacts

After training, you'll have:

| File | Size | Description |
|------|------|-------------|
| `model.pkl` | ~10KB | Trained Logistic Regression classifier |
| `vectorizer.pkl` | ~500KB-2MB | TF-IDF vectorizer with vocabulary |

These files are required for making predictions.

---

## 🔄 Retraining the Model

To retrain with new data:

1. Update `data/train.csv` with new reviews
2. Delete old model files:
   ```bash
   del model.pkl vectorizer.pkl
   ```
3. Run training again:
   ```bash
   python -m training.train_model
   ```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'training'"

**Solution:** Run scripts from the project root directory:
```bash
cd d:\Projects\AI-Based-Review-Classifier
python -m training.train_model
```

### Issue: "FileNotFoundError: data/train.csv not found"

**Solution:** Ensure your CSV file is in the correct location:
```
AI-Based-Review-Classifier/
└── data/
    └── train.csv
```

### Issue: Low accuracy (<70%)

**Possible causes:**
- Insufficient training data
- Imbalanced dataset
- Poor quality text data (too much noise)

**Solutions:**
- Collect more training samples (aim for 1000+ samples)
- Balance positive/negative samples
- Clean text data (remove HTML, special characters)

### Issue: "MemoryError" during training

**Solution:** Reduce vocabulary size:
```python
vectorizer = TfidfVectorizer(max_features=5000)
```

---

## 📈 Performance Optimization

### Profile Training Time
```bash
python profile_training.py
```

### Reduce Model Size
- Limit TF-IDF features: `max_features=3000`
- Use `sparse` matrices (already implemented)
- Consider model compression after training

---

## 🎯 Next Steps

After successful training:

1. ✅ Evaluate model performance on test data
2. ✅ Deploy API to production (see `readme.md`)
3. ✅ Set up continuous integration for retraining
4. ✅ Monitor model performance in production
5. ✅ Collect feedback and retrain periodically

---

## 📚 Additional Resources

- [Scikit-learn Logistic Regression Docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [TF-IDF Vectorizer Guide](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Flask API Documentation](https://flask.palletsprojects.com/)

---

## 🆘 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify all dependencies are installed
4. Ensure data files are in correct format

---

**Happy Training! 🎉**
