import pandas as pd
from pathlib import Path

def load_data(path) -> pd.DataFrame:
    '''
    Function that loads data into a dataframe
    '''
    df = pd.read_csv(path, parse_dates=['datum'])
    df = df.set_index('datum')
    ## datetime if needed meta
    df.index.rename('Date', inplace=True)
    return df

# df = load_data('data/raw/extracted/salesdaily.csv')
# print(df.head())
