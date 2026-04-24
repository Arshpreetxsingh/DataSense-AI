# DataSense AI 🚀

End-to-end Machine Learning system for **Customer Churn Prediction** with an interactive dashboard built using Streamlit.

---

## 🎯 Problem

Businesses often struggle to identify customers who are likely to leave (churn).
Losing customers directly impacts revenue and growth.

---

## 💡 Solution

DataSense AI provides a complete pipeline that:

* Cleans and preprocesses raw customer data
* Applies feature engineering for better insights
* Trains multiple machine learning models
* Selects the best model using ROC-AUC
* Provides real-time churn prediction through an interactive UI

---

## 🧠 Features

* ✅ End-to-end ML pipeline (data → model → prediction)
* ✅ Automated preprocessing (handles numeric + categorical data)
* ✅ Feature engineering (tenure groups, spending behavior, etc.)
* ✅ Multiple model training (Logistic Regression, Random Forest, XGBoost)
* ✅ Model selection using ROC-AUC
* ✅ Interactive Streamlit dashboard
* ✅ Real-time prediction system

---

## 📊 Results

* Best Model: Random Forest *(example — update after your run)*
* ROC-AUC: 0.84 *(replace with your score)*
* Accuracy: 0.80 *(replace with your score)*

---

## 📸 Demo

>Screenshots of your Streamlit app here

images/demo.png


---

## 🏗️ Project Structure

```
DataSense-AI/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── eda.ipynb
│   ├── experimentation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── predict.py
│
├── app/
│   └── app.py
│
├── utils/
│   └── helper.py
│
├── models/
├── config.yaml
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

Clone the repository:

```
git clone https://github.com/Arshpreetxsingh/DataSense-AI.git
cd DataSense-AI
```

Create virtual environment:

```
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🚀 Run the Project

### 1. Train the model

```
python test.py
```

### 2. Run the Streamlit app

```
streamlit run app/app.py
```

---

## 🔍 How It Works

1. Raw dataset is loaded and cleaned
2. Feature engineering is applied
3. Data is split into training and testing sets
4. Multiple models are trained
5. Best model is selected based on ROC-AUC
6. Model is saved and used for prediction
7. User interacts with the app to get predictions

---

## 🧠 Key Learnings

* Built a production-style ML pipeline
* Handled real-world data preprocessing challenges
* Designed modular and scalable architecture
* Integrated ML models into a user-facing application
* Implemented model evaluation and selection

---

## 🔮 Future Improvements

* Add explainability (Why a customer churns)
* Deploy the app online
* Add support for multiple datasets
* Improve UI/UX

---

## 🛠️ Tech Stack

* Python
* pandas, numpy
* scikit-learn, XGBoost
* Streamlit

---

## 👤 Author

**Arshpreet Singh**

GitHub: https://github.com/Arshpreetxsingh
