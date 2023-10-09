# standard library imports
# third-party imports
# local imports
# from naive_bayes import NaiveBayes
from file_mod import print_to_file
from naive_bayes import NaiveBayes
from matrix import Matrix

def main():
    testNaiveBayes()

def testNaiveBayes():
    filepath = "data/flowers-dataset.csv"
    class_name = "iris"
    train_percentage = 70
    test_percentage = 30
    separator = '|'
    
    # create naive bayes classifier
    naive_bayes = NaiveBayes(class_name, filepath, separator)
    
    # fit the model
    model = naive_bayes.fit( 
        train_percentage, 
        test_percentage
    )
    
    print_to_file('a', {'Calculo de probabilidad por clase normalizado': model})
    
    # evaluate the model
    results_df, success, failure = naive_bayes.evaluate()
    print_to_file('a', {'Evaluacion final': results_df, 'success' : success, 'failure' : failure})
    
    # save results to csv
    naive_bayes.results_csv(results_df)
    Matrix()

if __name__ == '__main__':
    main()