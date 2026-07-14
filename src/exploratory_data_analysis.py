"""
This module performs Exploratory Data Analysis (EDA)
on the Apple stock price dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """
    Load the dataset.
    """
    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df[
        (df["Date"] >= "2010-01-01") &
        (df["Date"] <= "2017-12-31")
    ]

    if "OpenInt" in df.columns:
        df.drop(columns=["OpenInt"], inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def closing_price_plot(df):
    """
    Plot Apple closing price.
    """

    plt.figure(figsize=(12,5))

    plt.plot(
        df["Date"],
        df["Close"],
        color="blue",
        linewidth=2
    )

    plt.title("Apple Closing Price (2010-2017)")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")

    plt.grid(alpha=0.3)

    plt.show()


def correlation_heatmap(df):
    """
    Plot correlation matrix.
    """

    plt.figure(figsize=(8,6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Matrix")

    plt.show()


def box_plot(df):
    """
    Detect outliers using boxplot.
    """

    plt.figure(figsize=(8,4))

    sns.boxplot(x=df["Close"])

    plt.title("Outlier Detection")

    plt.show()


if __name__ == "__main__":

    DATA_PATH = "data/raw/aapl.us.txt"

    df = load_data(DATA_PATH)

    closing_price_plot(df)

    correlation_heatmap(df)

    box_plot(df)