import pandas as pd
from DTAE import DTAE
import arff
from sklearn.metrics import roc_auc_score


def main():
    train_file = open('OCC Categorical Datasets/audiology/audiology.training5.arff', "r")
    train_dataset = arff.load(train_file)
    train_file.close()
    feature_names = list()
    for attribute in train_dataset['attributes']:
        feature_names.append(attribute[0])

    train_dataset = pd.DataFrame(train_dataset['data'], columns=feature_names)

    test_file = open('OCC Categorical Datasets/audiology/audiology.testing5.arff', "r")
    test_dataset = pd.DataFrame(arff.load(test_file)['data'], columns=feature_names)
    test_file.close()

    
    X_train = train_dataset.iloc[:, 0:train_dataset.shape[1]-1]
    y_train =  train_dataset.iloc[:, train_dataset.shape[1]-1 : train_dataset.shape[1]]

    X_test = test_dataset.iloc[:,  0:test_dataset.shape[1]-1]
    y_test = test_dataset.iloc[:, test_dataset.shape[1]-1 : test_dataset.shape[1]]

    dtae = DTAE()
    dtae.train(X_train, y_train)

    results = dtae.classify(X_test)

    auc = roc_auc_score(y_test, results)

    print(f"AUC =  {auc}")
    print(1 - auc)

    '''print(Style.RESET_ALL)
    print("-----------------------------------")
    dtae.classify_and_interpret(X_test.values[1])
    print("-----------------------------------")
    dtae.classify_and_interpret(X_test.values[5])
    print("-----------------------------------")
    dtae.classify_and_interpret(X_test.values[10])'''







main()