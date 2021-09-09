import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import time

RANDOM_SEED = 1337


def split(vectorized_train, labels):
    X = pd.read_pickle(vectorized_train).iloc[:, -5:]
    y = pd.read_pickle(labels)
    print(X.shape, y.shape)
    # print(X.head())
    # print(y.head())
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    print(X_train.shape, y_train.shape, X_dev.shape, y_dev.shape)
    return X_train, y_train, X_dev, y_dev


def train(X_train, y_train):
    t0 = time.time()
    clf = MLPClassifier(random_state=RANDOM_SEED, verbose=1, solver='lbfgs', activation='relu', learning_rate='adaptive',
                        hidden_layer_sizes=(4,3),max_iter=1000,early_stopping=True)
    clf.fit(X_train, y_train)
    time_to_train = time.time() - t0

    return clf, time_to_train


def metrics(clf, X_dev, y_dev):
    y_pred = clf.predict(X_dev)

    conf_mat = confusion_matrix(y_dev, y_pred)
    f1 = f1_score(y_dev, y_pred)
    accuracy = accuracy_score(y_dev, y_pred)
    return conf_mat, f1, accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'vectorized_training_data_file', help='file containing vectorized training data')
    parser.add_argument(
        'labels', help='file containing labels')
    parser.add_argument(
        'nn_model_output_file', help='file to contain trained nn model')
    parser.add_argument(
        'nn_metrics_file', help='file to contain trained nn model metrics on WikiTrain.csv')
    args = parser.parse_args()
    print("nn: Splitting data...")
    X_train, y_train, X_dev, y_dev = split(args.vectorized_training_data_file, args.labels)
    print("nn: Training model...")
    clf, time_to_train = train(X_train, y_train)
    print("nn: Getting metrics...")
    conf_mat, f1, accuracy = metrics(clf, X_dev, y_dev)

    print("nn: Writing results...")
    pickle.dump(clf, open(args.nn_model_output_file, 'wb'))
    metrics_dict = {'accuracy': accuracy, 'f1': f1, 'time_to_train': time_to_train}
    with open(args.nn_metrics_file, 'w') as metrics_file:
        json.dump(metrics_dict, metrics_file)
    # with open(args.logreg_metrics_file, 'w') as metrics_file:
    #     metrics_file.write('\n'.join([
    #         "Metrics for baseline model. ",
    #         "Use these as criteria of success for future models. ",
    #         "Scores higher than the below indicate an improvement over the baseline.",
    #         "Confusion Matrix",
    #         "(https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html):",
    #         str(conf_mat),
    #         "Area Under the Receiver Operating Characteristic Curve",
    #         "(https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html):",
    #         str(roc_auc),
    #         "F1 Score (Weighted Average of Precision and Recall [Summary/\"Grade\" of Confusion Matrix])",
    #         "(https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html):",
    #         str(f1),
    #         "Accuracy Score",
    #         "(https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html):",
    #         str(accuracy)
    #     ]))
