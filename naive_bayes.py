# standard library imports
# third-party imports
import pandas as pd
import numpy as np
from collections import defaultdict
# local imports
from file_mod import print_to_file
import clean_data as cd
import common_statistics as stats

class NaiveBayes:
    def fit(self, filepath, class_name, train_percentage, test_percentage, 
            separator=','):
        """Classify the dataset using the Naive Bayes algorithm.
        
        Args:
            filepath (string): filepath to the dataset
            class_name (string): name of the class column
            train_percentage (float): percentage of the dataset to be used for 
            training
            test_percentage (float): percentage of the dataset to be used for 
            testing
            separator (string): separator of the dataset
            
        Returns:
            tuple (pair, dict): ((success_rate, failure_rate), model): Verosimilitude 
            table of the attributes
        
        """
        df = cd.process_data(filepath, separator)
        train_df, test_df = cd.divide_data(df, train_percentage, test_percentage) 
        verisim_table = self.compute_verisimilitude(train_df, class_name)  
        prob_per_class = self.compute_prob_density(test_df, verisim_table, class_name)
        normalized_prob_df = self.compute_normalized_prob_density(prob_per_class)
        return test_df, normalized_prob_df
        
    def compute_verisimilitude(self, train_df, class_name):
        """Get verisimilitude table of the data set.
        
        Args:
            train_df (DataFrame): training set
            class_name (string): name of the class column
            
        Returns:
            dict: model of the Naive Bayes classifier
        
        """
        verisimilitude_table = defaultdict(dict)
        
        for attribute in train_df.columns:
            if attribute != class_name:
                if pd.api.types.is_numeric_dtype(train_df[attribute]):
                    versim_table = stats.compute_versim_table_num(train_df, class_name, attribute)
                    print_to_file('a', {'attribute' : attribute, 'verosim' : versim_table})
                    verisimilitude_table[attribute] = versim_table
                else:
                    freq_table = stats.compute_freq_table_cat(train_df, class_name, attribute)
                    versim_table = stats.compute_versim_table_cat(freq_table)
                    print_to_file('a', {'attribute' : attribute,'freq' : freq_table,'verosim' : versim_table})
                    verisimilitude_table[attribute] = versim_table
        
        class_freq = stats.compute_freq_table_class(train_df, class_name)
        print_to_file('a', {'class' : class_name, 'freq' : class_freq})
        verisimilitude_table[class_name] = class_freq
        
        return verisimilitude_table
    
    def compute_prob_density(self, test_df, verisim_table, class_name):
        """Get probabilities per class.
        
        Args:
            verisimilitude_table (dict): verisimilitude table of the data set
            class_name (string): name of the class column
            
        Returns:
            dict: probabilities per class
        
        """
        prob_df = pd.DataFrame()
        
        for class_value in test_df[class_name].unique():
            self.compute_prob_class(test_df, verisim_table, class_value, class_name, prob_df)
        
        print_to_file('a', {'Calculo de probabilidad por clase': prob_df})
        
        return prob_df
    
    def compute_prob_class(self, test_df, verisim_table, class_value, class_name, prob_df):
        """Get probabilities per class.
        
        Args:
            verisimilitude_table (dict): verisimilitude table of the data set
            class_name (string): name of the class column
            class_value (string): value of the class
            
        Returns:
            dict: probabilities per class
        
        """
        def calculate_probability(instance):
            probability = 1.0
            
            for attribute in test_df.columns:
                if attribute != class_name and not pd.api.types.is_numeric_dtype(test_df[attribute]):
                    attr_value = instance[attribute]
                    if attr_value in verisim_table[attribute].columns:
                        probability *= verisim_table[attribute].loc[class_value, attr_value]
                    
                elif attribute == class_name:
                    probability *= verisim_table[class_name][class_value]
                    
                elif attribute != class_name and pd.api.types.is_numeric_dtype(test_df[attribute]):
                    attr_value = instance[attribute] 
                    mean = verisim_table[attribute].loc[class_value, 'mean']
                    std = verisim_table[attribute].loc[class_value, 'std']
                    probability *= stats.compute_prob_density_func(attr_value, mean, std)
                    
            return probability
        
        prob_df[class_value] = test_df.apply(calculate_probability, axis=1)
        
    def compute_normalized_prob_density(self, prob_df):
        """Get the normalized probabilities.
        
        Args:
            prob_df (DataFrame): probabilities per class
            
        Returns:
            DataFrame: normalized probabilities
        
        """
        def format_percentage(value):
            return round(value * 100, 2)
            
        normalized_prob_df = prob_df.div(prob_df.sum(axis = 1), axis = 0)\
                                    .applymap(format_percentage)
        
        print_to_file('a', {'Calculo de probabilidad por clase normalizado': normalized_prob_df})
        
        return normalized_prob_df
        
    def evaluate(self, test_df, normalized_prob_df, class_name):
        results_df = pd.DataFrame()
        results_df['real'] = test_df[class_name]
        results_df['predicted'] = normalized_prob_df.idxmax(axis=1)
        results_df['correct'] = results_df.apply(lambda row: row['real'] == row['predicted'], axis=1)
        
        success = (results_df['correct'].value_counts()[True]) / len(results_df) * 100
        failure = 100.0 - success
        
        print_to_file('a', {'Evaluacion final': results_df, 'success' : success, 'failure' : failure})
        
        return results_df, success, failure 