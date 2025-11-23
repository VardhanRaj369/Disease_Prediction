# 🩺 Disease Prediction & Medicine Recommender App  
A Machine Learning + Streamlit project that predicts diseases based on user-selected symptoms and recommends the most suitable medicine.

---

## 🚀 Overview  
This project uses a **Random Forest Classifier** trained on a medical symptoms dataset to predict one of 42 diseases.  
Based on the predicted disease, the app recommends the most relevant **medicine** using a predefined mapping.

The entire application is built using:

- **Python**
- **Machine Learning (Scikit-Learn)**
- **Streamlit (Web UI)**
- **Pandas & NumPy**
- **JSON medicine mapping**

The app runs **fully online** and requires no installation for end-users.

---

## 🎯 Features
- ✔ Select multiple symptoms from a dropdown  
- ✔ ML model predicts the most likely disease  
- ✔ App recommends the best medicine  
- ✔ Clean & interactive Streamlit user interface  
- ✔ Lightweight, fast, and beginner-friendly  
- ✔ Easy deployment via Streamlit Cloud  

---

## 🧠 Dataset  
The project uses the *Disease Prediction Using Machine Learning* dataset from Kaggle, containing:

- **132 symptoms** as binary features  
- **1 prognosis column** representing the disease  
- Training and testing CSV files included in `data/` folder

---

## 🛠 Project Structure

Medical_Project/
│
├── app.py # Streamlit app
├── medicine_map.json # Disease → Medicine mapping
├── requirements.txt # Dependencies
│
├── data/
│ ├── Training.csv
│ └── Testing.csv
│
├── model/
│ └── disease_model.pkl # Trained ML model
│
└── scripts/
└── train_model.py # Training script


---

