"""
Predict the next trading day's closing price.
"""

import joblib

from src.model_training import (
    prepare_dataset,
    train_test_split_time
)


def load_best_model():
    """Load the trained model."""
    return joblib.load("models/best_stock_price_model.pkl")


def predict_next_day(model, X_test):
    """
    Predict the next trading day's closing price.
    """

    latest_features = X_test.iloc[[-1]]

    prediction = model.predict(latest_features)

    print("\nPredicted Next Trading Day Closing Price")
    print("---------------------------------------")
    print(f"${prediction[0]:.2f}")

    return prediction


if __name__ == "__main__":

    DATA_PATH = "data/raw/aapl.us.txt"

    df = prepare_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split_time(df)

    model = load_best_model()

    predict_next_day(model, X_test)