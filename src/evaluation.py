"""
Evaluate the best trained stock price prediction model.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.model_training import prepare_dataset, train_test_split_time


def load_model():
    """
    Load the trained model.
    """
    return joblib.load("models/best_stock_price_model.pkl")


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and save metrics.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    evaluation = pd.DataFrame({
        "Metric": [
            "Mean Absolute Error",
            "Root Mean Squared Error",
            "R² Score"
        ],
        "Value": [
            round(mae, 3),
            round(rmse, 3),
            round(r2, 4)
        ]
    })

    evaluation.to_csv(
        "results/evaluation_metrics.csv",
        index=False
    )

    print("\nEvaluation Metrics\n")
    print(evaluation)

    return predictions


def actual_vs_predicted(y_test, predictions):
    """
    Plot Actual vs Predicted prices.
    """

    plt.figure(figsize=(12,5))

    plt.plot(
        y_test.values,
        label="Actual Price"
    )

    plt.plot(
        predictions,
        label="Predicted Price"
    )

    plt.title("Actual vs Predicted Stock Prices")
    plt.xlabel("Test Samples")
    plt.ylabel("Closing Price")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.show()


def scatter_plot(y_test, predictions):
    """
    Scatter plot.
    """

    plt.figure(figsize=(7,7))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

    plt.title("Actual vs Predicted Scatter Plot")

    plt.grid(alpha=0.3)

    plt.show()


def residual_plot(y_test, predictions):
    """
    Residual distribution.
    """

    residuals = y_test - predictions

    plt.figure(figsize=(10,5))

    plt.hist(
        residuals,
        bins=30
    )

    plt.title("Residual Error Distribution")

    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")

    plt.grid(alpha=0.3)

    plt.show()


if __name__ == "__main__":

    DATA_PATH = "data/raw/aapl.us.txt"

    df = prepare_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split_time(df)

    model = load_model()

    predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    actual_vs_predicted(
        y_test,
        predictions
    )

    scatter_plot(
        y_test,
        predictions
    )

    residual_plot(
        y_test,
        predictions
    )