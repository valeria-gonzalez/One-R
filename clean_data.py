import pandas as pd
from file_mod import print_to_file

def process_data(filepath, separator):
    df = pd.read_csv(filepath,sep=separator) 
    df = df.sample(frac=1).reset_index(drop=True) # shuffle the dataset
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def divide_data(df, train_percentage, test_percentage):
    df_len = len(df) # get the length of the dataset
    train_len = round(df_len * (train_percentage / 100))
    train_df = df[:train_len] # get the training set
    
    if (test_percentage == 100):
        test_df = df[:]
    else: 
        test_df = df[train_len:]
    
    print_to_file('w', {'Training dataset': train_df, 
                        'Test dataset': test_df}
                  )
    return train_df, test_df
    
    