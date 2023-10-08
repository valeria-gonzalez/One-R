import pandas as pd
import numpy as np

def compute_freq_table_cat(train_df, class_name, attribute_name):
    """Compute frequency table of a categorical attribute given the class with 
        Laplace correction. X axis = attribute, y axis = class.

    Args:
        train_df (dataframe): training set
        class_name (string): name of the class column
        attribute_name (string): name of the attribute column

    Returns:
        dataframe: frequency table of the attribute given the class
    """
    freq_table = pd.crosstab(train_df[class_name], train_df[attribute_name]) 
    freq_table = freq_table.apply(lambda cell: cell + 1) # add 1 to each cell
    return freq_table

def compute_versim_table_cat(freq_table): 
    """Compute verisimilitude table of a categorical attribute given the class.
       For each row in the frequency table, normalizes the row.

    Args:
        freq_table (dataframe): frequency table of the attribute given the class
    """
    def normalize_column(row):
        return row / row.sum() # divide each cell by the sum of the row
    
    normalized_freq_table = freq_table.apply(normalize_column, axis=1)
    return normalized_freq_table

def compute_versim_table_num(train_df, class_name, attribute_name):
    """Compute verisimilitude table of a numerical attribute given the class.
       Groups values of attribute by class and calculates mean and std.

    Args:
        train_df (dataframe): training set
        class_name (string): name of the class column
        attribute_name (string): name of the attribute column

    Returns:
        dataframe: frequency table of the attribute given the class
    """
    freq_table = train_df.groupby(class_name)[attribute_name].agg(['mean', 'std'])
    return freq_table

def compute_freq_table_class(train_df, class_name):
    """Compute frequency table of the class values.
       Counts the ocurrences of a class value in the training set
       and divides it by the total number of instances.

    Args:
        train_df (dataframe): training set
        class_name (string): name of the class column

    Returns:
        dataframe: frequency table of the class values
    """
    freq_table = train_df[class_name].value_counts() # counts ocurrences of each class value
    no_instances = len(train_df) 
    normalized_freq_table = freq_table.div(no_instances) # divide each cell by the no. instances
    return normalized_freq_table

def compute_prob_density_func(attribute_value, class_mean, class_std):
   """Computes the probability density function of the attribute value given
      a class.
   
   Args:
       attribute_value (int): attribute value
       mean_std_tuple (dict): mean_std_tuple
       
   Returns:
       float: density function of the attribute value
   """
   if class_std == 0:
       return 0 # avoid division by 0
   exponent = np.exp(-((attribute_value-class_mean)**2 / (2*class_std**2)))
   return exponent / (np.sqrt(2*np.pi) * class_std)