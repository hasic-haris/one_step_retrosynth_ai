import pandas as pd


def read_dataset(path):
    return pd.read_csv(path + "data_processed.csv")