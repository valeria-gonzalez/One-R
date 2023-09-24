# One-R algorithm

Implementation of One Rule classification method on a set of random data.

### Installation
```
pip install requirements.txt
python main.py
```

### Usage
```
    from one_r import OneR # import module

    one_r = OneR() # create class instance

    # excute whole algorithm
    test_data, model = one_r.classify_data(
        filepath, 
        class_name, 
        train_percentage, 
        test_percentage
    )

    # obtain only model for a given dataframe
    model = one_r.get_model(train_df, class_name)

    # obtain only frequency table for a given attribute
    frequency_table = one_r.get_frequency_table(
        attribute_values, 
        class_values
    )

    # obtain best rule given a frequency table
    best_rule = one_r.get_attribute_rule(frequency_table)

    # test dataframe given a model
    success_rate, failure_rate = one_r.test_data(
        test_df, 
        model, 
        class_name
    )
```
### Authored by
Valeria Gonzalez Segura 217441582 (INNI)

Jonhathan Jacob Higuera Camacho 221351806 (INNI)

Diego Tristán Domínguez Dueñas 217812629 (INNI)

Minería de datos D01

Profesor Israel Román Godínez

2023B
