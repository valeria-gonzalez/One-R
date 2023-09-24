# standard library imports
# third-party imports
import pandas as pd
from collections import defaultdict
# local imports
from file_mod import print_to_file

class OneR:
    def classify_data(self, filepath, class_name, train_percentage, test_percentage):
        """Classify the dataset using the OneR algorithm.
        
        Args:
            filepath (string): filepath to the dataset
            class_name (string): name of the class column
            train_percentage (float): percentage of the dataset to be used for training
            test_percentage (float): percentage of the dataset to be used for testing
            
        Returns:
            tuple (success_rate, model): Instance of OneR classifier
        
        """
        df = pd.read_csv(filepath) # read the dataset
        df_copy = df.copy() # create a copy of the dataset
        df_copy = df_copy.sample(frac=1).reset_index(drop=True) # shuffle the dataset
        
        # remove leading and trailing whitespaces from the dataset
        df_copy = df_copy.map(lambda x: x.strip() if isinstance(x, str) else x)
        
        df_len = len(df_copy) # get the length of the dataset
        
        # get the amount of instances for the training set
        train_len = round(df_len * (train_percentage / 100))
        
        train_df = df_copy[:train_len] # get the training set
        # print(f"Training dataset: \n\n{train_df}\n")
        
        # get the amount of instances for the test set
        test_df = df_copy[:] if (test_percentage == 100) else df_copy[train_len:]
        # print(f"Test dataset: \n\n{test_df}\n")
        
        print_to_file('w', {'Training dataset': train_df, 
                            'Test dataset': test_df}
                      )
        
        model = self.get_model(train_df, class_name) # get the model
        success_rate = self.test_data(test_df, model, class_name) # test the dataset
        
        return success_rate, model
        
    def get_model(self, train_df, class_name):
        """Get the model of the OneR classifier.
        
        Args:
            train_df (DataFrame): training set
            class_name (string): name of the class column
            
        Returns:
            dict: model of the OneR classifier
        
        """
        rules = [] # list of dict of rules for each attribute
        INF = float('inf')
        best_rule = (INF, -1, "") # (min_tot_error, index in list, attribute_name)
        
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
                
                # get the rules for the attribute and the frequency table
                attribute_rule = self.get_attribute_rule(frequency_table)
                
                rules.append(attribute_rule) # append the rules to the list
                
                # update rule with the lowest total error
                if attribute_rule['Total_error'] < best_rule[0]:
                    best_rule = (attribute_rule['Total_error'], len(rules) - 1, attribute)
                    
        best_rule_index = best_rule[1] # get the index of the best rule
        model = rules[best_rule_index] # get the best dict rule
        model['Attribute'] = best_rule[2] # add the attribute name to the model
        
        return model
    
    def get_frequency_table(self, attribute_values, class_values):
        """Get the frequency table of the attribute.
        
        Args:
            attribute_values (list): list of attribute values
            class_values (list): list of class values
            
        Returns:
            dict: frequency table of the attribute
        
        """
        frequency_table = defaultdict(lambda: defaultdict(int))
        paired_values = zip(attribute_values, class_values)
        
        for attribute_value, class_value in paired_values:
            # increment the frequency of the attribute value and class value
            frequency_table[attribute_value][class_value] += 1 
            # record amount of instances of the attribute value
            frequency_table[attribute_value]['Total'] += 1 
            
        return frequency_table
    
    def get_attribute_rule(self, frequency_table):
        """Get the rules for the attribute based on its frequency table.
        
        Args:
            frequency_table (dict): frequency table of the attribute
            
        Returns:
            dict: rules for the attribute based on its frequency table
        
        """
        attribute_rule = dict()
        total_error_num = 0
        total_error_denom = 0
        
        for attribute, class_dictionary in frequency_table.items():
            # get the total number of instances of the attribute and delete key
            total = class_dictionary.pop("Total", None) 
            total_error_denom += total # add to the total error denominator
            
            # get the class value with the highest frequency
            max_class = max(class_dictionary, key=class_dictionary.get)
            
            # create rule that determines class for the attribute
            attribute_rule[attribute] = max_class 
            
            class_dictionary.pop(max_class, None) # delete key
            
            # add to total error numerator the sum of frequency of other class values
            total_error_num += sum(class_dictionary.values()) 
        
        attribute_rule['Total_error'] = total_error_num / total_error_denom
        
        return attribute_rule
                    
    def test_data(self, test_df, model, class_name):
        """Test the dataset using the model of the OneR classifier.
        
        Args:
            test_df (DataFrame): test set
            model (dict): model of the OneR classifier
            class_name (string): name of the class column
            
        Returns:
            tuple (success_rate, failure_rate): success rate of the test set
        
        """
        model_attribute = model['Attribute'] # get attribute name from the model
        attribute_values = list(test_df[model_attribute]) # get the attribute values
        class_values = list(test_df[class_name]) # get the class values
        # pair the values (attribute, class)
        paired_values = zip(attribute_values, class_values) 
        
        success_rate = 0 # number of correct predictions
        failure_rate = 0 # number of incorrect predictions
        
        for attribute_value, class_value in paired_values:
            # get the predicted class value
            predicted_class = model[attribute_value] 
            
            # compare the predicted class value with the actual class value
            if predicted_class == class_value:
                success_rate += 1
            else:
                failure_rate += 1
                
        len_test_df = len(test_df) # get the length of the test set
        success_rate = (success_rate * 100) / len_test_df # calculate success rate
        failure_rate = (failure_rate * 100) / len_test_df # calculate failure rate
        
        return success_rate, failure_rate
        