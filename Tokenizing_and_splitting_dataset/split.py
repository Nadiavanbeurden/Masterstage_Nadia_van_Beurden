import pandas as pd 
import numpy as np
from typing import Tuple
import argparse 

parser = argparse.ArgumentParser(description='Binning data')
parser.add_argument('-n_bins', type=int, default=2, help='Number of bins (= Vocabulary size)')
parser.add_argument('-dataset', type=str, default='Signal', help='Signal or Background')

args = parser.parse_args()
n_bins = args.n_bins
dataset = args.dataset

try:
    assert dataset in ['Background', 'Signal']
except AssertionError:
    print("Please enter 'Background' or 'Signal'")

# Read dataframe to split
df = pd.read_csv(f'binned_dataset.csv')

def train_test_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame]: 
    train, test, val = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)),int(0.9*len(df))])
    return train, test, val

# Shuffle data to remove potential row ordering
df = df.sample(frac=1)

# Split data into train, test and validation set
df_train, df_test, df_val = train_test_val_split(df)

# Store dataframes as csv
df_train.to_csv(f'binned_trainset.csv', index=False) 
df_test.to_csv(f'binned_testset.csv', index=False) 
df_val.to_csv(f'binned_validationset.csv', index=False)