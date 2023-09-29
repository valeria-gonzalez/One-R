# standard library imports
# third-party imports
# local imports
from one_r import OneR
from naive_bayes import NaiveBayes
from file_mod import print_to_file
        
def main():
    testNaiveBayes()

def testNaiveBayes():
    # filepath = "data/golf-dataset-numerical.csv"
    filepath = "data/flowers-dataset.csv"
    class_name = "iris"
    train_percentage = 70
    test_percentage = 30
    
    naive_bayes = NaiveBayes()
    
    test_data, model = naive_bayes.classify_data( # ((acierto, desacierto), dict)
        filepath, 
        class_name, 
        train_percentage, 
        test_percentage
    )
    
    # success_rate, failure_rate = test_data
    
    # print_to_file('a', {'Success rate': success_rate, 
    #                     'Failure rate': failure_rate, 
    #                     'Resulting model': model}
    #               )


def testOneR():
    filepath = "data/golf-dataset-categorical.csv"
    class_name = "Play"
    train_percentage = 70
    test_percentage = 30
    
    one_r = OneR()
    
    test_data, model = one_r.classify_data( # ((acierto, desacierto), dict)
        filepath, 
        class_name, 
        train_percentage, 
        test_percentage
    )
    
    success_rate, failure_rate = test_data
    
    print_to_file('a', {'Success rate': success_rate, 
                        'Failure rate': failure_rate, 
                        'Resulting model': model}
                  )

if __name__ == '__main__':
    main()