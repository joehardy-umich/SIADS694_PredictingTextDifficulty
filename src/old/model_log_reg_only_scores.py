import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import time
import numpy as np
from tqdm import tqdm

RANDOM_SEED = 1337

# def create_score_interactions()

limit_to = [
    'mean_age',
    'median_age',
    'max_age',
    'mean_perc_known_lem',
    'median_perc_known_lem',
    'mean_freq_pm',
    'median_freq_pm',
    'count_uncommon',
    'perc_uncommon',
    'mean_concreteness',
    'min_concreteness',
    'mean_perc_known',
    'min_perc_known',
    'mean_syllables',
    'max_syllables',
    'min_syllables',
    'num_words'
]


def split(vectorized_train, labels):
    X = pd.read_pickle(vectorized_train).loc[:, limit_to].astype(pd.SparseDtype('float', 0.))
    for col in tqdm(X.columns):
        if str(X[col].dtype) == 'Sparse[float64, 0]' or str(X[col].dtype) == 'Sparse[float64, 0.0]':
            X[col] = X[col].sparse.to_dense()
            X.loc[X[col] < 0, col] = 0.
            X.loc[pd.isnull(X[col]), col] = 0.
            X[col] = pd.arrays.SparseArray(X[col])

    ss = StandardScaler()
    X = ss.fit_transform(X)
    y = pd.read_pickle(labels)
    print(X.shape, y.shape)
    # print(X.head())
    # print(y.head())
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    print(X_train.shape, y_train.shape, X_dev.shape, y_dev.shape)
    return X_train, y_train, X_dev, y_dev


def train(X_train, y_train):
    t0 = time.time()
    clf = LogisticRegression(random_state=RANDOM_SEED, solver='lbfgs', verbose=1,max_iter=2000)
    clf.fit(X_train, y_train)
    time_to_train = time.time() - t0

    return clf, time_to_train


def metrics(clf, X_dev, y_dev):
    y_pred = clf.predict(X_dev)

    conf_mat = confusion_matrix(y_dev, y_pred)
    roc_auc = roc_auc_score(y_dev, clf.predict_proba(X_dev)[:, 1])
    f1 = f1_score(y_dev, y_pred)
    accuracy = accuracy_score(y_dev, y_pred)
    print(clf.__dict__)
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
    print("Log Reg (Only Scores): Splitting data...")
    X_train, y_train, X_dev, y_dev = split(args.vectorized_training_data_file, args.labels)
    print("Log Reg (Only Scores): Training model...")
    clf, time_to_train = train(X_train, y_train)
    print("Log Reg (Only Scores): Getting metrics...")
    conf_mat, roc_auc, f1, accuracy = metrics(clf, X_dev, y_dev)

    print("Log Reg (Only Scores): Writing results...")
    pickle.dump(clf, open(args.logreg_model_output_file, 'wb'))
    metrics_dict = {'accuracy': accuracy, 'roc_auc': roc_auc, 'f1': f1, 'time_to_train': time_to_train}
    print(metrics_dict)
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
