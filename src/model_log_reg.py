import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import time
import numpy as np

RANDOM_SEED = 1337


def split(vectorized_train, labels,subset=1000000):
    print("Reading data...")
    X = pd.read_pickle(vectorized_train)
    #X[pd.isnull(X)]=0.
    y = pd.read_pickle(labels)
    # p = PCA(n_components=20)
    # X=p.fit_transform(X)
    # print("Subsetting data...")
    # random_indices = np.random.choice(range(len(X)), subset, replace=False)
    # X = X.iloc[random_indices]
    # y = y.iloc[random_indices]
    print(X.shape, y.shape)
    # print(X.head())
    # print(y.head())
    print("Performing split...")
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    print(X_train.shape, y_train.shape, X_dev.shape, y_dev.shape)
    return X_train, y_train, X_dev, y_dev


def train(X_train, y_train):
    t0 = time.time()
    clf = LogisticRegression(random_state=RANDOM_SEED, solver='lbfgs', verbose=1,max_iter=5000)
    clf.fit(X_train, y_train)
    time_to_train = time.time() - t0

    return clf, time_to_train


def metrics(clf, X_dev, y_dev):
    y_pred = clf.predict(X_dev)

    conf_mat = confusion_matrix(y_dev, y_pred)
    roc_auc = roc_auc_score(y_dev, clf.predict_proba(X_dev)[:, 1])
    f1 = f1_score(y_dev, y_pred)
    accuracy = accuracy_score(y_dev, y_pred)
    return conf_mat, roc_auc, f1, accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'vectorized_training_data_file', help='file containing vectorized training data')
    parser.add_argument(
        'labels', help='file containing labels')
    parser.add_argument(
        'logreg_model_output_file', help='file to contain trained log reg model')
    parser.add_argument(
        'logreg_metrics_file', help='file to contain trained log reg model metrics on WikiTrain.csv')
    args = parser.parse_args()
    print("Log Reg: Splitting data...")
    X_train, y_train, X_dev, y_dev = split(args.vectorized_training_data_file, args.labels)
    print("Log Reg: Training model...")
    clf, time_to_train = train(X_train, y_train)
    print("Log Reg: Getting metrics...")
    conf_mat, roc_auc, f1, accuracy = metrics(clf, X_dev, y_dev)

    print("Log Reg: Writing results...")
    pickle.dump(clf, open(args.logreg_model_output_file, 'wb'))
    metrics_dict = {'accuracy': accuracy, 'roc_auc': roc_auc, 'f1': f1, 'time_to_train': time_to_train}
    with open(args.logreg_metrics_file, 'w') as metrics_file:
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
