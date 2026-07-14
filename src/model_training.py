"""
Train and compare machine learning models
for Apple stock price prediction.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import (
    add_technical_indicators,
    create_lag_features,
    create_daily_return,
    create_target,
    remove_missing_values
)


def prepare_dataset(file_path):
    """Load data and generate all features."""

    df = load_data(file_path)
    df = preprocess_data(df)

    df = add_technical_indicators(df)
    df = create_lag_features(df)
    df = create_daily_return(df)
    df = create_target(df)
    df = remove_missing_values(df)

    return df


def train_test_split_time(df):
    """Split the dataset using a time-based split."""

    train_data = df[df["Date"] < "2016-01-01"]
    test_data = df[df["Date"] >= "2016-01-01"]

    X_train = train_data.drop(columns=["Date", "Target"])
    y_train = train_data["Target"]

    X_test = test_data.drop(columns=["Date", "Target"])
    y_test = test_data["Target"]

    return X_train, X_test, y_train, y_test


def get_models():
    """Return all models used in the project."""

    return {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42,
            max_depth=10
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
    }


def train_models(X_train, X_test, y_train, y_test):
    """Train all models and compare performance."""

    models = get_models()

    results = []
    trained_models = {}

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append([name, mae, rmse, r2])

        trained_models[name] = model

    results_df = pd.DataFrame(
        results,
        columns=["Model", "MAE", "RMSE", "R2 Score"]
    )

    results_df = results_df.sort_values(
        by="R2 Score",
        ascending=False
    ).reset_index(drop=True)

    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    results_df.to_csv("results/model_comparison.csv", index=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, "models/best_stock_price_model.pkl")

    print("\nBest Model:", best_model_name)

    return results_df, best_model


if __name__ == "__main__":

    DATA_PATH = "data/raw/aapl.us.txt"

    df = prepare_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split_time(df)

    results_df, best_model = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(results_df)