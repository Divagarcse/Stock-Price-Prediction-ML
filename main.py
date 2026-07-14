"""
Run the entire stock price prediction pipeline.
"""

from src.model_training import (
    prepare_dataset,
    train_test_split_time,
    train_models
)

from src.evaluation import (
    load_model,
    evaluate_model
)

from src.prediction import predict_next_day


def main():

    DATA_PATH = "data/raw/aapl.us.txt"

    print("=" * 60)
    print("Stock Price Prediction using Machine Learning")
    print("=" * 60)

    # Prepare dataset
    df = prepare_dataset(DATA_PATH)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split_time(df)

    # Train models
    results, best_model = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nModel Comparison")
    print(results)

    # Evaluate best model
    predictions = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    # Predict next day's price
    predict_next_day(best_model, X_test)


if __name__ == "__main__":
    main()