import pandas as pd
import numpy as np

def compute_freq_table_cat(train_df, class_name, attribute_name):
    # create frequency table given the class and an attribute
    freq_table = pd.crosstab(train_df[class_name], train_df[attribute_name]) 
    freq_table = freq_table.apply(lambda cell: cell + 1) # add 1 to each cell
    return freq_table

def compute_versim_table_cat(freq_table): 
    def normalize_column(row):
        return row / row.sum() # divide each cell by the sum of the row
    normalized_freq_table = freq_table.apply(normalize_column, axis=1)
    return normalized_freq_table

def compute_versim_table_num(train_df, class_name, attribute_name):
    freq_table = train_df.groupby(class_name)[attribute_name].agg(['mean', 'std'])
    return freq_table

def compute_freq_table_class(train_df, class_name):
    freq_table = train_df[class_name].value_counts()
    total_sum = freq_table.sum()
    normalized_freq_table = freq_table.div(total_sum)
    return normalized_freq_table

def compute_prob_density_func(attribute_value, mean, std):
   """Get the density function of the attribute value.
   
   Args:
       attribute_value (int): attribute value
       mean_std_tuple (dict): mean_std_tuple
       
   Returns:
       float: density function of the attribute value
   
   """
   if std == 0:
       return 0
   exponent = np.exp(-(((attribute_value - mean) ** 2) / (2 * (std ** 2))))
   return exponent / (np.sqrt(2 * np.pi) * std)