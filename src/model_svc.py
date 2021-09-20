import json

import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import time
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler

RANDOM_SEED = 1337

limit_to = [
    'mean_age',
    'median_age',
    'max_age',
    'mean_perc_known_lem',  #
    'median_perc_known_lem',  #
    'mean_freq_pm',
    'median_freq_pm',
    'count_uncommon',
    'perc_uncommon',
    'mean_concreteness',
    'min_concreteness',
    'mean_perc_known',
    'min_perc_known',  #
    'mean_syllables',
    'max_syllables',
    'min_syllables',
    'num_words'
]


def split(vectorized_train, labels, subset=1000000):
    print("Reading data...")
    X = pd.read_pickle(vectorized_train).astype(pd.SparseDtype('float', 0.))
    # X = X.drop(columns='median_perc_known_lem')
    # x_fis = [0.0777801, 0.06242877, 0.07475293, 0.02180517, 0.06721678, 0.0633527, 0.06518147, 0.038125, 0.01709966,
    #          0.01260322, 0.06371432, 0.02656552, 0.04307288, 0.02682482, 0.01116191, 0.0473337, 0.10916434, 0.17181673]
    # X = X.iloc[:, [i for i, c in enumerate(x_fis) if c > 0.05]]
    # col_sum = None
    # for col in tqdm(X.columns):
    #     if str(X[col].dtype) == 'Sparse[float64, 0]' or str(X[col].dtype) == 'Sparse[float64, 0.0]':
    #         X[col] = X[col].sparse.to_dense()
    #         X.loc[X[col] < 0, col] = 0.
    #         X.loc[pd.isnull(X[col]), col] = 0.
    #         if col not in limit_to:
    #             if col_sum is not None:
    #                 col_sum += X[col]
    #             else:
    #                 col_sum = X[col]
    #             X = X.drop(columns=col)
    #         # X[col] = pd.arrays.SparseArray(X[col])
    #     else:
    #         if col not in limit_to:
    #             if col_sum is not None:
    #                 col_sum += X[col]
    #             else:
    #                 col_sum = X[col]
    #             X = X.drop(columns=col)
    scaler = RobustScaler()
    pca = PCA(n_components=5)
    X = pd.concat([X,pd.DataFrame(pca.fit_transform(X))],axis=1)
    X = scaler.fit_transform(X)
    # X[pd.isnull(X)]=0.
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
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    time_to_train = time.time() - t0

    return clf, time_to_train


def metrics(clf, X_dev, y_dev):
    y_pred = clf.predict(X_dev)

    conf_mat = confusion_matrix(y_dev, y_pred)
    #roc_auc = roc_auc_score(y_dev, clf.predict_proba(X_dev)[:, 1])
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
        'SVC_model_output_file', help='file to contain trained SVC model')
    parser.add_argument(
        'SVC_metrics_file', help='file to contain trained SVC model metrics on WikiTrain.csv')
    args = parser.parse_args()
    print("SVC: Splitting data...")
    X_train, y_train, X_dev, y_dev = split(args.vectorized_training_data_file, args.labels)
    print("SVC: Training model...")
    clf, time_to_train = train(X_train, y_train)
    print("SVC: Getting metrics...")
    conf_mat, f1, accuracy = metrics(clf, X_dev, y_dev)

    print("SVC: Writing results...")
    pickle.dump(clf, open(args.SVC_model_output_file, 'wb'))
    metrics_dict = {'accuracy': accuracy, 'f1': f1, 'time_to_train': time_to_train}
    print(metrics_dict)
    with open(args.SVC_metrics_file, 'w') as metrics_file:
        json.dump(metrics_dict, metrics_file)
    # with open(args.SVC_metrics_file, 'w') as metrics_file:
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
