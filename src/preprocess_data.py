import pandas as pd
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import timedelta


names_dictionary = {
    "M01AB" : "Sales volume of anti-inflammatory and antirheumatic products, non-steroids, Acetic acid derivatives and related ",
    "M01AE" : "Sales volume of anti-inflammatory and antirheumatic products, non-steroids, Propionic acid derivatives",
    "N02BA" : "Sales volume of other analgesics and antipyretics, Salicylic acid and derivatives",
    "N02BE" : "Sales volume of other analgesics and antipyretics, Pyrazolones and Anilides",
    "N05B" : "Sales volume of psycholeptics drugs, Anxiolytic drugs",
    "N05C" : "Sales volume of psycholeptics drugs, Hypnotics and sedatives drugs",
    "R03" : "Sales volume of drugs for obstructive airway diseases",
    "R06" : "Sales volume of antihistamines for systemic use",
}

# def single_ts_choice(df: pd.DataFrame) -> pd.Series:
#     print("Available drug categories:")
#     for code, desc in names_dictionary.items():
#         print(f"{code}: {desc}")
#     while True:
#         choice = input("\nEnter drug code (e.g. M01AB): ").strip().upper()
#         if choice in df.columns:
#             return df[choice]
#         else:
#             print(f"Invalid choice. Available options are: {list(names_dictionary.keys())}")

def single_ts_choice(df: pd.DataFrame) -> pd.Series:
    return df["M01AE"].copy()

def reset_indexer(ts) -> pd.DataFrame:
    prophet_ts = ts.reset_index()
    prophet_ts.columns = ['ds', 'y']
    return prophet_ts

def split_time_series(df, train_ratio=0.9):
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    return train_df, test_df
