import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from file_mod import print_to_file

def Matrix():
    df = pd.read_csv('results.csv')

    y_true = df['real']
    y_pred = df['predicted']

    nombres_clases = ["Setosa", "Versicolor", "Virginica"]

    confusion = confusion_matrix(y_true, y_pred)

    precision = precision_score(y_true, y_pred, average=None)

    recall = recall_score(y_true, y_pred, average=None)

    accuracy = accuracy_score(y_true, y_pred)

    matriz = pd.DataFrame(confusion, columns=nombres_clases, index=nombres_clases)
    print_to_file('a', {'Matriz de confusion': matriz})

    precision_clase = dict(zip(nombres_clases, precision))
    print_to_file('a', {'Presicion': precision_clase})

    recall_clase = dict(zip(nombres_clases, recall))
    print_to_file('a', {'Recall': recall_clase})

    print_to_file('a', {'Exactitud del modelo': accuracy})