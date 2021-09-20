import json

import pandas as pd
from sklearn.naive_bayes import ComplementNB as nb
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pickle
import time
import numpy as np
from collections import Counter
from tqdm import tqdm

RANDOM_SEED = 1337


def split(vectorized_train, labels, subset=-1):#int(416000*0.8)):
    print("Reading data...")
    X = pd.read_pickle(vectorized_train).astype(pd.SparseDtype('float', 0.))
    print(X.shape,X.columns)
    # for col in tqdm(X.columns):
    #     if str(X[col].dtype) == 'Sparse[float64, 0]' or str(X[col].dtype) == 'Sparse[float64, 0.0]':
    #         X[col] = X[col].sparse.to_dense()
    #         X.loc[X[col] < 0, col] = 0.
    #         X.loc[pd.isnull(X[col]), col] = 0.
    #         X[col] = pd.arrays.SparseArray(X[col])

    # print(Counter(X.dtypes.values))

    y = pd.read_pickle(labels)
    # p = PCA(n_components=20)
    # X = p.fit_transform(X)
    # print(labels)
    if subset > 0:
        print("Subsetting data...")
        random_indices = np.random.choice(range(len(X)), subset, replace=False)
        X_old = X
        y_old = y
        X = X.iloc[random_indices]
        y = y.iloc[random_indices]
    # print(X.shape, y.shape)
    # print(X.head())
    # print(y.head())
    print("Assigning split...")
    X_train, X_dev, y_train, y_dev = train_test_split(np.arange(0, X.shape[0]), np.arange(0, y.shape[0]), test_size=0.2,
                                                      random_state=RANDOM_SEED)
    print("Performing split...")
    X_train = X.iloc[X_train]
    # print(X_train.describe())
    X_dev = X.iloc[X_dev]
    y_train = y.iloc[y_train]
    y_dev = y.iloc[y_dev]

    print(X_dev)
    if subset > 0:
        X_dev = pd.concat([X_dev, X_old.iloc[np.invert(random_indices)]])
        y_dev = pd.concat([y_dev, y_old.iloc[np.invert(random_indices)]])
    print(X_train.shape, y_train.shape, X_dev.shape, y_dev.shape)
    return X_train, y_train, X_dev, y_dev


def train(X_train, y_train):
    t0 = time.time()
    clf = nb(class_prior=[0.5, 0.5])  # random_state=RANDOM_SEED, solver='lbfgs', verbose=1)
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
        'NB_model_output_file', help='file to contain trained NB model')
    parser.add_argument(
        'NB_metrics_file', help='file to contain trained NB model metrics on WikiTrain.csv')
    args = parser.parse_args()
    print("NB: Splitting data...")
    X_train, y_train, X_dev, y_dev = split(args.vectorized_training_data_file, args.labels)
    print("NB: Training model...")
    clf, time_to_train = train(X_train, y_train)
    print("NB: Getting metrics...")
    conf_mat, roc_auc, f1, accuracy = metrics(clf, X_dev, y_dev)

    print("NB: Writing results...")
    pickle.dump(clf, open(args.NB_model_output_file, 'wb'))
    metrics_dict = {'accuracy': accuracy, 'roc_auc': roc_auc, 'f1': f1, 'time_to_train': time_to_train}
    print(metrics_dict)
    with open(args.NB_metrics_file, 'w') as metrics_file:
        json.dump(metrics_dict, metrics_file)
    # with open(args.NB_metrics_file, 'w') as metrics_file:
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
