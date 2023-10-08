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
    def __init__(self, class_name, filepath, separator = ','):
        """Naive Bayes classifier.

        Args:
            filepath (string): filepath to the dataset csv.
            separator (str, optional): Separator used in file. Defaults to ','.
        """
        self.class_name = class_name
        self.filepath = filepath
        self.separator = separator
        self.train_df = None
        self.test_df = None
        self.normalized_prob_density = None
        
    def fit(self, train_percentage, test_percentage):
        """Classify the dataset using the Naive Bayes algorithm.
           Obtains the normalized probability of each class being assigned to 
           each instance in the training dataset. It divides the dataset, obtains
           verisimilitude table and computes the probability density function.
        
        Args:
            class_name (string): name of the class column
            train_percentage (float): training data percentage
            test_percentage (float): test data percentage
            
        Returns:
            tuple(dataframe, dataframe): test dataset and normalized probabilities
        """
        df = cd.process_data(self.filepath, self.separator)
        
        self.train_df, self.test_df = cd.divide_data(df, train_percentage, test_percentage) 
        print_to_file('w', {'Training dataset': self.train_df, 'Test dataset': self.test_df})
        
        verisim_table = self.compute_verisimilitude(self.train_df, self.class_name)  
        
        prob_per_class = self.compute_prob_density(self.test_df, verisim_table, self.class_name)
        print_to_file('a', {'Calculo de probabilidad por clase': prob_per_class})
        
        self.normalized_prob_density = self.compute_normalized_prob_density(prob_per_class)
        
        return self.normalized_prob_density
        
    def compute_verisimilitude(self, train_df, class_name):
        """Get verisimilitude table of the training data set.
            Computes verisimilitude table for each attribute given a class.
        
        Args:
            train_df (DataFrame): training set
            class_name (string): name of the class column
            
        Returns:
            dict: versimilitude table of the data set
        """
        verisimilitude_table = defaultdict(dict)
        
        for attribute in train_df.columns:
            if attribute != class_name:
                # for numerical attributes
                if pd.api.types.is_numeric_dtype(train_df[attribute]):
                    versim_table = stats.compute_versim_table_num(train_df, class_name, attribute)
                    print_to_file('a', {'attribute num' : attribute, 'verosim' : versim_table})
                    verisimilitude_table[attribute] = versim_table
                
                # for categorical attributes
                else:
                    freq_table = stats.compute_freq_table_cat(train_df, class_name, attribute)
                    versim_table = stats.compute_versim_table_cat(freq_table)
                    print_to_file('a', {'attribute cat' : attribute,'freq' : freq_table,'verosim' : versim_table})
                    verisimilitude_table[attribute] = versim_table
        
        # for class attribute
        class_freq = stats.compute_freq_table_class(train_df, class_name)
        print_to_file('a', {'class' : class_name, 'freq' : class_freq})
        verisimilitude_table[class_name] = class_freq
        
        return verisimilitude_table
    
    def compute_prob_density(self, test_df, verisim_table, class_name):
        """Get probabilities of each class for every instance.
        
        Args:
            verisimilitude_table (dict): verisimilitude table of the data set
            class_name (string): name of the class column
            
        Returns:
            dict: probabilities per class
        
        """
        prob_df = pd.DataFrame()
        
        for class_value in test_df[class_name].unique():
            self.compute_prob_class(test_df, verisim_table, class_value, class_name, prob_df)
        
        return prob_df
    
    def compute_prob_class(self, test_df, verisim_table, class_value, class_name, prob_df):
        """Get probabilities for every instance given a specific class.
        
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
                # for categorical attributes
                if attribute != class_name and not pd.api.types.is_numeric_dtype(test_df[attribute]):
                    attr_value = instance[attribute]
                    if attr_value in verisim_table[attribute].columns:
                        probability *= verisim_table[attribute].loc[class_value, attr_value]
                
                # for class attribute
                elif attribute == class_name:
                    probability *= verisim_table[class_name][class_value]
                
                # for numerical attributes
                elif attribute != class_name and pd.api.types.is_numeric_dtype(test_df[attribute]):
                    attr_value = instance[attribute] 
                    mean = verisim_table[attribute].loc[class_value, 'mean']
                    std = verisim_table[attribute].loc[class_value, 'std']
                    # calculate probability density function
                    probability *= stats.compute_prob_density_func(attr_value, mean, std)
                    
            return probability
        
        prob_df[class_value] = test_df.apply(calculate_probability, axis=1)
        
    def compute_normalized_prob_density(self, prob_df):
        """Normalize the probabilities of each class for every instance.
           For every instance, divides each class probability by the sum of 
           all class probabilities.
        
        Args:
            prob_df (DataFrame): probabilities per class
            
        Returns:
            DataFrame: normalized probabilities
        
        """
        def format_percentage(value):
            return round(value * 100, 2) # format to 2 decimals percentage
            
        norm_prob_df = prob_df.div(prob_df.sum(axis = 1), axis = 0)\
                                    .applymap(format_percentage)
        
        return norm_prob_df
        
    def evaluate(self):
        """Evaluate the test dataset given the normalized probabilities of each class.
            Compares the real class with the predicted class (class with greatest
            normalized percentage) and calculates the success and failure percentages.

        Returns:
            tuple(dataframe, success rate, failure rate): dataframe with real class, 
            predicted class, correct prediction; success and failure percentages
        """
        results_df = pd.DataFrame()
        results_df['real'] = self.test_df[self.class_name]
        results_df['predicted'] = self.normalized_prob_density.idxmax(axis=1)
        results_df['correct'] = results_df.apply(lambda row: row['real'] == row['predicted'], axis=1)
        
        success = (results_df['correct'].value_counts()[True]) / len(results_df) * 100
        failure = 100.0 - success
        
        return results_df, success, failure 
    
    def results_csv(self, results_df):
        """Save the results dataframe to a csv file.
            Creates a csv file with the real class and the predicted class.

        Args:
            results_df (dataframe): resulting dataframe of the evaluation
        """
        results_df[['real', 'predicted']].to_csv('results.csv', index = False)