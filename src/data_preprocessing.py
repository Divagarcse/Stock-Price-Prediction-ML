"""
This module loads and preprocesses the Apple stock price dataset.
"""

import pandas as pd


def load_data(file_path):
    """
    Load the stock price dataset.

    Parameters:
        file_path (str): Path to dataset.

    Returns:
        pd.DataFrame
    """
    df = pd.read_csv(file_path)

    print("Dataset Loaded Successfully")
    print("Shape:", df.shape)

    return df


def preprocess_data(df):
    """
    Perform basic preprocessing.

    Steps:
    - Convert Date column to datetime
    - Keep data from 2010 to 2017
    - Remove unnecessary columns
    - Reset index

    Returns:
        pd.DataFrame
    """

    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Keep data between 2010 and 2017
    df = df[
        (df["Date"] >= "2010-01-01") &
        (df["Date"] <= "2017-12-31")
    ]

    # Drop Open Interest column if present
    if "OpenInt" in df.columns:
        df = df.drop(columns=["OpenInt"])

    # Reset index
    df.reset_index(drop=True, inplace=True)

    print("\nData Preprocessing Completed")
    print("Final Shape:", df.shape)

    return df


def dataset_information(df):
    """
    Display dataset information.
    """

    print("\nDataset Information")
    print("-" * 40)

    print(df.info())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nStatistical Summary")
    print(df.describe())


if __name__ == "__main__":

    DATA_PATH = "data/raw/aapl.us.txt"

    df = load_data(DATA_PATH)

    df = preprocess_data(df)

    dataset_information(df)

    print("\nFirst Five Records")
    print(df.head())