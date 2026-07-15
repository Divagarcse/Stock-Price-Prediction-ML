# 📈 Stock Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts the next trading day's closing price of Apple Inc. using historical stock market data and machine learning regression algorithms.

The project demonstrates a complete end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, model evaluation, and future price prediction.

---

## 🎯 Objective

To build a machine learning model that accurately predicts the next trading day's closing price of Apple Inc. using historical stock market data.

---

## 📂 Dataset

- **Dataset:** Apple Inc. Historical Stock Prices
- **Source:** Kaggle (Stocks Dataset)
- **Company:** Apple Inc. (AAPL)

Features include:

- Date
- Open
- High
- Low
- Close
- Volume

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- TA (Technical Analysis Library)
- Joblib

---

## ⚙️ Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Preprocessing
6. Feature Scaling
7. Model Training
8. Model Evaluation
9. Model Comparison
10. Save Best Model
11. Next-Day Stock Price Prediction

---

## 🤖 Machine Learning Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

---

## 📊 Model Performance

| Model | R² Score |
|-------|---------:|
| Linear Regression | **0.9959** |
| XGBoost | 0.4661 |
| Random Forest | 0.4468 |
| Decision Tree | 0.4190 |

### Best Model

**Linear Regression**

Evaluation Metrics:

- MAE: **1.09**
- RMSE: **1.57**
- R² Score: **0.9959**

---

## 📁 Project Structure

```plaintext
Stock-Price-Prediction-ML/
│
├── data/
│   └── raw/
│       └── aapl.us.txt
│
├── models/
│   └── best_stock_price_model.pkl
│
├── notebooks/
│   └── Stock_Price_Prediction_ML_Internship.ipynb
│
├── results/
│   ├── evaluation_metrics.csv
│   └── model_comparison.csv
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── exploratory_data_analysis.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── prediction.py
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```
## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Divagarcse/Stock-Price-Prediction-ML.git
```

### 2. Navigate to the project directory

```bash
cd Stock-Price-Prediction-ML
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Python project

```bash
python main.py
```

### 5. Explore the complete workflow (Optional)

If you want to understand how the model was built, open the notebook:

```text
notebooks/Stock_Price_Prediction_ML_Internship.ipynb
```

Run all cells sequentially to:

- Load the historical Apple stock dataset
- Perform data preprocessing and cleaning
- Conduct exploratory data analysis (EDA)
- Generate technical indicators and lag features
- Train multiple machine learning models
- Compare model performance
- Evaluate the best-performing model
- Predict the next trading day's closing price


### 6. Run Individual Modules 

Each module can be executed independently for testing and understanding the complete machine learning pipeline:

```bash
python -m src.data_preprocessing
python -m src.exploratory_data_analysis
python -m src.feature_engineering
python -m src.model_training
python -m src.evaluation
python -m src.prediction
```
---

## 📌 Results

The Linear Regression model achieved the highest prediction accuracy and successfully predicted the next trading day's closing price with a high R² score of **0.9959**.

---

## 🔮 Future Improvements

- Use real-time stock market APIs.
- Build a Streamlit web application.
- Include news sentiment analysis.
- Train deep learning models such as LSTM.

---

## 👨‍💻 Author

**Divagar G**

Machine Learning Internship Project


