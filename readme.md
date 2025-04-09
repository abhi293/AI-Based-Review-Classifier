# 🎯 AI-Based Review Classifier

## 📌 Overview
The **AI-Based Review Classifier** is a machine learning-powered web application designed to classify customer reviews as **Positive, Negative, or Neutral**. The system integrates a **Flask-based AI model** with a **Node.js backend** and supports real-time classification.

---
## 🚀 Features
- ✅ **Machine Learning-Based Sentiment Analysis**
- ✅ **RESTful API for Classification**
- ✅ **Preprocessing & Model Training Scripts**
- ✅ **Flask API for AI Model Deployment**
- ✅ **Secure Node.js Backend**
- ✅ **Scalable & Modular Architecture**
- ✅ **Automated Training & Evaluation**
- ✅ **CI/CD & Deployment Support**

---
## 🏗️ System Architecture
```plaintext
User ➝ Frontend ➝ Backend (Node.js) ➝ AI Model (Flask) ➝ Database (MongoDB)
```

---
## 📂 Project Structure
```
(for %i in (file1.txt test.js ) do type nul > %i)    #To create multiple files at once
AI-Based-Review-Classifier/
│── 📂 ai_model/                  # AI Model (Python & Machine Learning)
│   ├── 📂 data/                    # Dataset storage
│   │   ├── train.csv          # Raw dataset
│   │   ├── test.csv
│   ├── 📂 training/                 # Training scripts
│   │   ├── preprocess.py           # Data preprocessing
│   │   ├── train_model.py          # Model training & saving
│   │   ├── evaluate.py             # Model evaluation
│   ├── 📂 api/                     # Flask API to expose model
│   │   ├── app.py                  # Flask API for predictions
│   │   ├── model_loader.py         # Model loading utility
│   │   ├── requirements.txt        # Python dependencies
│   ├── 📂 notebooks/               # Jupyter notebooks for experiments
│   │   ├── DataExploration.ipynb   # Dataset exploration
│   │   ├── ModelTesting.ipynb      # Model evaluation
│   ├── 📂 tests/                   # Model testing scripts
│   │   ├── test_predictions.py     # API testing for model predictions
│   │   ├── test_preprocessing.py   # Test data preprocessing steps
│   ├── ai_service.py               # Python client to call Flask API
│
└── README.md                      # Project documentation


```

---
## 🛠️ Tech Stack
### **Backend (API & Server)**
- **Node.js**, **Express.js**, **MongoDB**, **Socket.io**
- **Nodemailer** (Email notifications)

### **AI Model**
- **Python**, **Flask**, **Scikit-Learn**, **TF-IDF**
- **Jupyter Notebooks** (Experiments)

### **Frontend (Optional)**
- **React.js**, **TailwindCSS**

---
## 📡 API Endpoints
### **1️⃣ AI Model API (Flask)**
| Method | Endpoint          | Description              |
|--------|------------------|--------------------------|
| `POST` | `/predict`       | Classifies review       |

### **2️⃣ Backend API (Node.js)**
| Method | Endpoint                | Description                         |
|--------|------------------------|-------------------------------------|
| `POST` | `/api/ai/classify`      | Classifies user review            |
| `GET`  | `/api/reviews`         | Fetches all classified reviews    |

---
## ⚙️ Installation & Setup
### **1️⃣ Clone Repository**
```sh
git clone https://github.com/yourusername/AI-Based-Review-Classifier.git
cd AI-Based-Review-Classifier
```

### **2️⃣ Install Dependencies**
#### **Backend (Node.js)**
```sh
cd backend
npm install
```
#### **AI Model (Python)**
```sh
cd ai_model/api
pip install -r requirements.txt
```

### **3️⃣ Setup Environment Variables**
Create a `.env` file in the **backend/** folder:
```env
MONGO_URI=your_mongo_db_connection_string
PORT=5000
```

---
## 🚀 Running the Application
### **1️⃣ Start Backend (Node.js)**
```sh
cd backend
npm start
```

### **2️⃣ Start AI Model API (Flask)**
```sh
cd ai_model/api
python app.py
```

### **3️⃣ Test in Postman**
```
POST http://localhost:5000/api/ai/classify
Body: { "review": "This product is amazing!" }
```

---
## 🛠️ Deployment
### **Backend Deployment**
- Deploy on **Heroku**, **AWS**, or **Vercel**
- Use **PM2** for process management

### **AI Model Deployment**
- Deploy Flask API on **Render** or **Google Cloud**
- Use **Docker** for containerization

---
## 🔐 Security Enhancements
- **JWT Authentication**
- **CORS Handling**
- **Input Validation**

---
## 📊 Monitoring & Visualization
- **Grafana & Prometheus** for API monitoring
- **MLflow** for AI model tracking

---
## 🤝 Contributing
We welcome contributions! Follow these steps:
1. **Fork the repo**
2. **Create a new branch** (`feature-branch`)
3. **Commit your changes**
4. **Create a Pull Request**

---
## 📜 License
This project is licensed under the **MIT License**. Feel free to modify and use it!

🚀 **Happy Coding!** 🎉

