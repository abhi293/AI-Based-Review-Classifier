# 🎯 AI-Based-Review-Classifier

## 📌 Overview
The **AI-Based Review Classifier** is a Python-based machine learning application that classifies customer reviews as **Positive** or **Negative**. It uses a **Logistic Regression** model trained on **TF-IDF** features, and exposes a RESTful API for real-time predictions using **Flask** (with an example FastAPI implementation also included).

---
## 🚀 Features
- ✅ **Logistic Regression Model for Sentiment Analysis**
- ✅ **TF-IDF Text Vectorization**
- ✅ **RESTful API for Real-Time Classification (Flask)**
- ✅ **Preprocessing, Training, and Evaluation Scripts**
- ✅ **Jupyter Notebooks for Data Exploration and Testing**
- ✅ **Docker Support for Easy Deployment**
- ✅ **Automated Model Testing and Evaluation**

---
## 📂 Project Structure
```
AI-Based-Review-Classifier/
│   ai_service.py                # Python client for API
│   check_data_balance.py        # Data balance checker
│   Dockerfile                   # Docker deployment
│   evaluate.py                  # Model evaluation script
│   main.py                      # FastAPI example (optional)
│   model.pkl                    # Trained model
│   ModelTesting.py              # Model testing script
│   profile_training.py          # Profiling training time
│   sample.py                    # Sample model training
│   vectorizer.pkl               # Saved TF-IDF vectorizer
│   readme.md                    # Project documentation
│
├── api/
│   ├── app.py                   # Flask API for predictions
│   ├── model_loader.py          # Model/vectorizer loader
│   └── requirements.txt         # Python dependencies
│
├── notebooks/
│   ├── DataExploration.ipynb    # Data exploration
│   └── ModelTesting.ipynb       # Model evaluation
│
├── tests/
│   ├── test_predictions.py      # API prediction tests
│   └── test_preprocessing.py    # Preprocessing tests
│
└── training/
    ├── preprocess.py            # Data preprocessing
    ├── train_model.py           # Model training
    └── evaluate.py              # Model evaluation
```

---
## 🛠️ Tech Stack
- **Python 3.10+**
- **Flask** (REST API)
- **Scikit-Learn** (Logistic Regression, TF-IDF)
- **Pandas, NumPy**
- **Jupyter Notebooks** (Experiments)
- **Docker** (Deployment)

---
## 🤖 Model & Algorithm Details
- **Algorithm:** Logistic Regression (from scikit-learn)
- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Input:** Raw review text
- **Output:** Sentiment label (`positive` or `negative`) and confidence score
- **Model Artifacts:** `model.pkl` (Logistic Regression), `vectorizer.pkl` (TF-IDF)

---
## 📡 API Endpoints
### **Flask API**
| Method | Endpoint   | Description                |
|--------|------------|----------------------------|
| POST   | `/predict` | Classifies review text     |

**Sample Request:**
```
POST http://localhost:5000/predict
Content-Type: application/json
{
  "text": "This product is amazing!"
}
```
**Sample Response:**
```
{
  "prediction": "positive",
  "confidence": "98.00%"
}
```

---
## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```
git clone https://github.com/yourusername/AI-Based-Review-Classifier.git
cd AI-Based-Review-Classifier
```

### 2️⃣ Install Dependencies
```
pip install -r api/requirements.txt
```

---
## 🚀 Running the Application

### Start the Flask API
```
cd api
python app.py
```

The API will be available at `http://localhost:5000/predict`.

---
## 🐳 Docker Deployment
To build and run the app with Docker:
```
docker build -t review-classifier .
docker run -p 8001:8001 review-classifier
```

---
## 🧪 Testing
- Run `tests/test_predictions.py` to test API predictions.
- Run `tests/test_preprocessing.py` to test preprocessing.
- Use Jupyter notebooks in `notebooks/` for exploration and evaluation.

---
## 🤝 Contributing
We welcome contributions! Please fork the repo, create a branch, and submit a pull request.

---
## 📜 License
This project is licensed under the **MIT License**.

🚀 **Happy Coding!** 🎉

