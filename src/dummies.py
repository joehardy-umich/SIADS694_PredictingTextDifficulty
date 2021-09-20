import json
import time

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

RANDOM_SEED = 1337


def get_clfs():
    clf_dict = {}
    for strategy in {'stratified', 'most_frequent', 'prior', 'uniform'}:

        clf_dict[strategy] = {'clf': DummyClassifier(strategy=strategy)}
    return clf_dict


def split(vectorized_train, labels):
    X = pd.read_pickle(vectorized_train)#.iloc[:,-5:]
    y = pd.read_pickle(labels)
    print(X.shape, y.shape)
    # print(X.head())
    # print(y.head())
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    print(X_train.shape, y_train.shape, X_dev.shape, y_dev.shape)
    return X_train, y_train, X_dev, y_dev


def train(X_train, y_train, clfs):
    for strategy in clfs:
        clf = clfs[strategy]['clf']
        t0 = time.time()
        clf.fit(X_train, y_train)
        time_to_train = time.time() - t0
        clfs[strategy]['metrics'] = {}
        clfs[strategy]['metrics']['time_to_train'] = time_to_train

    return clfs


def metrics(clfs, X_dev, y_dev):
    for strategy in clfs:
        clf = clfs[strategy]['clf']
        y_pred = clf.predict(X_dev)

        clfs[strategy]['metrics']['conf_mat'] = confusion_matrix(y_dev, y_pred)
        clfs[strategy]['metrics']['roc_auc'] = roc_auc_score(y_dev, clf.predict_proba(X_dev)[:, 1])
        clfs[strategy]['metrics']['f1'] = f1_score(y_dev, y_pred)
        clfs[strategy]['metrics']['accuracy'] = accuracy_score(y_dev, y_pred)
    return clfs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'vectorized_training_data_file', help='file containing vectorized training data')
    parser.add_argument(
        'labels', help='file containing labels')
    parser.add_argument(
        'dummy_metrics_file', help='file to contain trained baseline model metrics on WikiTrain.csv')
    args = parser.parse_args()
    print("Dummy: Splitting data...")
    X_train, y_train, X_dev, y_dev = split(args.vectorized_training_data_file, args.labels)
    print("Dummy: Training model...")
    clfs = get_clfs()
    clfs = train(X_train, y_train, clfs)
    print("Dummy: Getting metrics...")
    clfs = metrics(clfs, X_dev, y_dev)

    print("Dummy: Writing results...")
    # pickle.dump(clf, open(args.dummy_model_output_file, 'wb'))
    metrics_dict = {
        strategy: {metric:clfs[strategy]['metrics'][metric] for metric in {'accuracy', 'roc_auc', 'f1', 'time_to_train'}} for
        strategy in
        clfs}  # {'accuracy': accuracy, 'roc_auc': roc_auc, 'f1': f1, 'time_to_train': time_to_train}
    with open(args.dummy_metrics_file, 'w') as metrics_file:
        json.dump(metrics_dict, metrics_file)
    # with open(args.baseline_metrics_file, 'w') as metrics_file:
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
