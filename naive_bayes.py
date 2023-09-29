# standard library imports
# third-party imports
import pandas as pd
import numpy as np
from collections import defaultdict
# local imports
from file_mod import print_to_file

class NaiveBayes:
    def classify_data(self, filepath, class_name, train_percentage, test_percentage, separator=','):
        """Classify the dataset using the OneR algorithm.
        
        Args:
            filepath (string): filepath to the dataset
            class_name (string): name of the class column
            train_percentage (float): percentage of the dataset to be used for training
            test_percentage (float): percentage of the dataset to be used for testing
            separator (string): separator of the dataset
            
        Returns:
            tuple (pair, dict): ((success_rate, failure_rate), model): Verosimilitude table of the attributes
        
        """
        df = pd.read_csv(filepath,sep=separator) # read the dataset
        df_copy = df.copy() # create a copy of the dataset
        df_copy = df_copy.sample(frac=1).reset_index(drop=True) # shuffle the dataset
        
        # remove leading and trailing whitespaces from the dataset
        df_copy = df_copy.map(lambda x: x.strip() if isinstance(x, str) else x)
        
        df_len = len(df_copy) # get the length of the dataset
        
        # get the amount of instances for the training set
        train_len = round(df_len * (train_percentage / 100))
        
        train_df = df_copy[:train_len] # get the training set
        print(f"Training dataset: \n\n{train_df}\n")
        
        # get the amount of instances for the test set
        test_df = df_copy[:] if (test_percentage == 100) else df_copy[train_len:]
        
        print_to_file('w', {'Training dataset': train_df, 
                            'Test dataset': test_df}
                      )
        print(f"Training dataset: \n\n{train_df}\n")
        print(f"Test dataset: \n\n{test_df}\n")
        
        model = self.get_model(train_df, class_name) # get the model
        success_rate = self.test_data(test_df, model, class_name) # test the dataset
        
        return success_rate, model
        
    def get_model(self, train_df, class_name):
        """Get the model of the Naive Bayes classifier.
        
        Args:
            train_df (DataFrame): training set
            class_name (string): name of the class column
            
        Returns:
            dict: model of the Naive Bayes classifier
        
        """
        probabilities = {} # dict = {[attribute]: {class: frequency/total, ...}, ...}
        
        attribute_names = list(train_df.columns) # get the attribute names
        class_values = list(train_df[class_name]) # get the class values
        
        for attribute in attribute_names:
            if attribute != class_name:
                # get the attribute values
                attribute_values = list(train_df[attribute])
                
                # get the frequency table of the attribute
                frequency_table = self.get_frequency_table(
                    attribute_values,
                    class_values
                )

                # get verosimilitude table of the attribute
                verosimilitude_table = self.get_verosimilitude_table(
                    frequency_table
                )
                
                probabilities[attribute] = verosimilitude_table
        
        probabilities[class_name] = self.get_probabilities_for_class(class_values)
        return probabilities

    def get_frequency_table(self, attribute_values, class_values):
        """Get the frequency table of the attribute.
        
        Args:
            attribute_values (list): list of attribute values
            class_values (list): list of class values
            
        Returns:
            dict: frequency table of the attribute
        
        """
        if type(attribute_values[0]) == int or type(attribute_values[0]) == float:
            return self.get_frecuency_table_numerical(attribute_values, class_values)
        
        frequency_table = defaultdict(lambda: defaultdict(int)) # [attribute]: {class: frequency, total: #}
        frequency_table = self.laplacian_correction(frequency_table, attribute_values, class_values) # Sum 1 to all
        paired_values = zip(attribute_values, class_values) # pair the values {(attribute, class), (attribute, class), ...}
        
        for attribute_value, class_value in paired_values:
            frequency_table[attribute_value][class_value] += 1 # frequency_table[sunny][yes] += 1
            frequency_table["Total"][class_value] += 1 # frequency_table[Total][yes] += 1
        
        return frequency_table
    
    def get_frecuency_table_numerical(self, attribute_values, class_values):
        """Get the frequency table of the numerical attribute.
        
        Args:
            attribute_values (list): list of attribute values
            class_values (list): list of class values
            
        Returns:
            dict: frequency table of the attribute (list of attribute values for each class)
        
        """
        frequency_table = defaultdict(lambda: list()) # [class]: [numerical attribute values]
        paired_values = zip(attribute_values, class_values) # pair the values {(attribute, class), (attribute, class), ...}
        
        for attribute_value, class_value in paired_values:
            frequency_table[class_value].append(attribute_value)
            
        return frequency_table
    
    def get_verosimilitude_table(self, frequency_table):
        """Get the verosimilitude table of the attribute.
        
        Args:
            frequency_table (dict): frequency table of the attribute
            
        Returns:
            dict: verosimilitude table of the attribute
        
        """
        if "Total" not in frequency_table: #It's a numerical attribute
            return self.get_verosimilitude_table_numerical(frequency_table)

        verosimilitude_table = defaultdict(lambda: defaultdict(float)) # [attribute]: {class: frequency/total}
        class_total = frequency_table.pop("Total", None) # class_total = {[yes] = #, [no] = #}
        for attribute, class_dictionary in frequency_table.items(): # attribute = sunny, overcast, rainy
            for clas, frequency in class_dictionary.items(): # class_dictionary = {[yes] = #, [no] = #}
                verosimilitude_table[attribute][clas] = frequency / class_total[clas]
        
        return verosimilitude_table

    def get_verosimilitude_table_numerical(self, frequency_table):
        """Get the verosimilitude table of the numerical attribute.
        
        Args:
            frequency_table (dict): frequency table of the attribute
            
        Returns:
            dict: verosimilitude table of the attribute
        
        """
        verosimilitude_table = defaultdict() # [class]: (mean, std)
        for class_value, attribute_values in frequency_table.items():
            verosimilitude_table[class_value] = (np.mean(attribute_values), np.std(attribute_values)) # attribute_values = list(23,1,43,51,2,66,343)
        return verosimilitude_table
    
    def laplacian_correction(self, frequency_table, attribute_values, class_values):
        """Sum one to all the values of the frequency table.
        
        Args:
            frequency_table (dict): frequency table of the attribute
            attribute_values (list): list of attribute values
            class_values (list): list of class values
        
        Returns:
            dict: frequency table of the attribute with one added to all values
        """
        unique_attribute_values = tuple(set(attribute_values))
        unique_class_values = tuple(set(class_values))
        for attribute_value in unique_attribute_values:
            for class_value in unique_class_values:
                frequency_table[attribute_value][class_value] += 1
                frequency_table["Total"][class_value] += 1
        return frequency_table
    
    
    def get_probabilities_for_class(self, class_values):
        """Get the probabilities for each class.
        
        Args:
            class_values (list): list of class values
            
        Returns:
            dict: probabilities for each class
        
        """
        probabilities_class = {}
        unique_class_values = tuple(set(class_values))
        for class_value in unique_class_values:
            probabilities_class[class_value] = class_values.count(class_value) / len(class_values)
        return probabilities_class

    def test_data(self, test_df, model, class_name):
        """Test the dataset using the model of the Naive Bayes classifier.
        
        Args:
            test_df (DataFrame): test set
            model (dict): model of the Naive Bayes classifier
            class_name (string): name of the class column
            
        Returns:
            tuple (success_rate, failure_rate): success rate of the test set
        
        """
        class_values = list(test_df[class_name]) # get the class values in train_df
        unique_class_values = tuple(set(class_values)) # get the unique class values
        
        success_rate = 0 # number of correct predictions
        failure_rate = 0 # number of incorrect predictions

        probability_dict = {}

        for indice, instancia in test_df.iterrows(): # iterate over the test set
            probability_dict[indice] = {} # probability_dict = {[indice]: {class: probability, ...}, ...}
            probability_dict[indice]["Total"] = 0 # initialize the total probability for normalization
            for class_value in unique_class_values: 
                probability_dict[indice][class_value] = 1 # probability_dict = {[indice]: {class: probability, ...}
                for attribute, attribute_values in model.items():
                    # attribute = sunny, overcast, rainy
                    # if categorical -> attribute_values = {sunny: {yes: probabity#, no: probabity#}, overcast: {yes: probabity#, no: probabity#}, ...}
                    # if numerical -> attribute_values = {yes: (mean, std), no: (mean, std), class: (mean, std)}
                    if class_value in attribute_values:
                        if attribute == class_name: # if the attribute is the class column
                            probability_dict[indice][class_value] *= attribute_values[class_value]
                        else: # if the attribute is not the class column (numerical)
                            probability_dict[indice][class_value] *= self.get_density_function(instancia[attribute], attribute_values[class_value])
                    else:
                        probability_dict[indice][class_value] *= attribute_values[instancia[attribute]][class_value]

                probability_dict[indice]["Total"] += probability_dict[indice][class_value]
        
        probability_dict_normalized = self.normalize_probability(probability_dict)

        for indice, instancia in test_df.iterrows(): # iterate over the test set
            max_probability = max(probability_dict_normalized[indice], key=probability_dict_normalized[indice].get)
            if instancia[class_name] == max_probability:
                success_rate += 1
            else:
                failure_rate += 1

        len_test_df = len(test_df) # get the length of the test set
        success_rate = (success_rate * 100) / len_test_df # calculate success rate
        failure_rate = (failure_rate * 100) / len_test_df # calculate failure rate
        
        return success_rate, failure_rate
    
    def normalize_probability(self, probability_dict):
        """Normalize the probability of each class.
        
        Args:
            probability_dict (dict): probability dictionary
            
        Returns:
            dict: probability dictionary normalized
        
        """
        probability_dict_normalized = {}
        for indice, probability in probability_dict.items(): # probability = {class: probability, ...}
            probability_dict_normalized[indice] = {}
            for class_value, probability_value in probability.items():
                if class_value != "Total":
                    probability_dict_normalized[indice][class_value] = (probability_value / probability["Total"]) * 100
        return probability_dict_normalized

    def get_density_function(self, attribute_value, mean_std_tuple):
        """Get the density function of the attribute value.
        
        Args:
            attribute_value (int): attribute value
            mean_std_tuple (dict): mean_std_tuple
            
        Returns:
            float: density function of the attribute value
        
        """
        mean, std = mean_std_tuple
        exponent = np.exp(-((attribute_value-mean)**2 / (2*std**2)))
        return exponent / (np.sqrt(2*np.pi) * std)