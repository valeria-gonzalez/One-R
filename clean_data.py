import pandas as pd
from file_mod import print_to_file

def process_data(filepath, separator):
    """Reads the dataset from csv file and returns a DataFrame.
       Shuffles the dataset and removes trailing whitespaces.

    Args:
        filepath (string): filepath of the dataset
        separator (string/char): separator of the dataset

    Returns:
        dataframe: dataset
    """
    df = pd.read_csv(filepath,sep=separator) 
    df = df.sample(frac=1).reset_index(drop=True) # shuffle the dataset
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def divide_data(df, train_percentage, test_percentage):
    """Divides the dataset into training and test sets given the percentages.
       The first x rows are used for training and the rest for testing.
    Args:
        df (dataframe): dataset 
        train_percentage (float): training percentage
        test_percentage (float): test percentage

    Returns:
        tuple(dataframe, dataframe): training and test datasets
    """
    df_len = len(df) 
    train_len = round(df_len * (train_percentage / 100))
    train_df = df[:train_len] # first train_len rows to train
    
    if (test_percentage == 100):
        test_df = df[:] # all the rows to test
    else: 
        test_df = df[train_len:] # rest of the rows to test
    
    return train_df, test_df
    
    