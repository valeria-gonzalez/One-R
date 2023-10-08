# standard library imports
# third-party imports
# local imports
# from naive_bayes import NaiveBayes
from file_mod import print_to_file
import json
from naive_bayes import NaiveBayes
        
def main():
    testNaiveBayes()

def testNaiveBayes():
    filepath = "data/flowers-dataset.csv"
    class_name = "iris"
    train_percentage = 70
    test_percentage = 30
    separator = '|'
    
    naive_bayes = NaiveBayes()
    
    test_df, model = naive_bayes.fit( 
        filepath, 
        class_name, 
        train_percentage, 
        test_percentage,
        separator
    )
    
    results_df, success, failure = naive_bayes.evaluate(test_df, model, class_name)
    results_df[['real', 'predicted']].to_csv('results.csv', index = False)

if __name__ == '__main__':
    main()