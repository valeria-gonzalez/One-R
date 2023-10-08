# Naive Bayes algorithm

Implementation of Naive Bayes classification method on the iris dataset.

## Installation
```
pip install requirements.txt
python main.py # will return output at output.txt
```

## Naive Bayes Class 
Naive Bayes Classifier module.

### Usage
```
from naive_bayes import NaiveBayes # import module

# create class instance
naive_bayes = NaiveBayes(class_name, filepath, separator) 

# fit the model
model = naive_bayes.fit( 
    train_percentage, 
    test_percentage
)

# evaluate the model
results_df, success, failure = naive_bayes.evaluate()

# optional generate csv with real class and predicted class
naive_bayes.result_csv(results_df)
```
### Extra methods

- Generate verisimilitude table of training dataset
```
verisim_table = naive_bayes.compute_verosimilitude(train_df, class_name)
```

- Compute probability of each class for every instance in the training dataset
```
prob_density_table = naive_bayes.compute_prob_density(test_df, verisim_table, class_name)
```

- Compute probability for an individual class for every instance in the training dataset
```
prob_for_class = pd.DataFrame()
naive_bayes.compute_prob_class(test_df, verisim_table, class_value, class_name, prob_for_class)
```

- Compute normalized probability table (percentage) given a probability table
```
normalized_prob_table = naive_bayes.compute_normalized_prob_density(prob_density_table)
```

## Common_statistics Module 
Module to perform common operations needed in classification algorithms, such as 
frequency tables, verisimilitude tables, probability density function.

### Usage

```
import common_statistics as stats
```

- Create frequency table of categorical values given a class

```
freq_table = stats.compute_freq_table_cat(train_df, class_name, attribute_name)
```

- Create a verisimilitude table for categorical values given a class
```
verisimilitude_table = stats.compute_versim_table_cat(freq_table)
```

- Create verisimilitude table for numerical attributes
```
verisimilitude_table = stats.compute_versim_table_num(train_df, class_name, attribute_name)
```

- Create frequency table for class values in the dataset
```
frequency_table = stats.compute_freq_table_class(train_df, class_name)
```

- Compute probability density function for a numerical attribute
```
prob_density_function = stats.compute_prob_density_func(attribute_value, class_mean, class_std)
```

## Clean Data Module
Module to process a dataframe before being given to a classification method.

### Usage

```
import clean_data as cd
```
- Read a dataset from csv file, shuffle its contents and strip spaces
```
clean_df = cd.process_data(filepath, separator)
```
- Divide dataset into training set and test set.

```
train_df, test_df = cd.divide_data(clean_df, train_percentage, test_percentage)
```


## Authored by
Valeria Gonzalez Segura 217441582 (INNI)

Jonhathan Jacob Higuera Camacho 221351806 (INNI)

Diego Tristán Domínguez Dueñas 217812629 (INNI)

Minería de datos D01

Profesor Israel Román Godínez

2023B
