import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
import pickle
import time
import numpy as np
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer as imputer

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


# def create_score_interactions()
def load(vectorized_test):#, imputer_, scaler):
    X = pd.read_pickle(vectorized_test)
    X[pd.isnull(X)] = 0.
    X = X.astype(pd.SparseDtype('float', 0.))
    X['sum_all'] = X['sum_1'] + X['sum_0'] + X['sum_none']
    X['sum_ratio'] = X['sum_1'] / (X['sum_0'] + X['sum_none'] + 1)
    X['sum_diff'] = X['sum_1'] - X['sum_0']
    # X = imputer_.transform(X)
    # X = scaler.transform(X)
    return X


def split(vectorized_train, labels):
    # X = pd.read_pickle(vectorized_train)
    X = pd.read_pickle(vectorized_train)
    X[pd.isnull(X)] = 0.
    X = X.astype(pd.SparseDtype('float', 0.))
    X['sum_all'] = X['sum_1'] + X['sum_0'] + X['sum_none']
    X['sum_ratio'] = X['sum_1'] / (X['sum_0'] + X['sum_none'] + 1)
    X['sum_diff'] = X['sum_1'] - X['sum_0']
    #print("Imputing...")
    #imputer_ = imputer(missing_values=-1, strategy='mean', add_indicator=False)
    # ix = np.random.choice(np.arange(0,X.shape[0]),size=10000)
    # imputer_.fit(X.iloc[ix,:])
    #X = imputer_.fit_transform(X)
    # pca = PCA(n_components=10)
    # X = pca.fit_transform(X)
    # X = pd.DataFrame(, columns=X.columns)
    # col_names = list(X.columns)
    print(X.shape)  # , X.columns)
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
    #
    # X['sentence_sum'] = col_sum
    # X.to_pickle('data/model/vectorized_summed.pkl')

    # X = X.drop(columns='median_perc_known_lem')
    # x_fis = [0.0777801, 0.06242877, 0.07475293, 0.02180517, 0.06721678, 0.0633527, 0.06518147, 0.038125, 0.01709966,
    #          0.01260322, 0.06371432, 0.02656552, 0.04307288, 0.02682482, 0.01116191, 0.0473337, 0.10916434, 0.17181673]
    # X = X.iloc[:, [i for i, c in enumerate(x_fis) if c > 0.05]]
    # X = X[['num_words', 'min_syllables', 'mean_syllables', 'count_uncommon', 'sentence_sum_eda','median_freq_pm']]
    # print(X.columns)
    y = pd.read_pickle(labels)

    #print(X.shape, y.shape)
    # y = y[(X!=0).all(1)].reset_index(drop=True)
    # X = X[(X!=0).all(1)].reset_index(drop=True)
    #scaler = StandardScaler()
    # pca = PCA(n_components=5, random_state=1337)
    # X = pd.concat([X, pd.DataFrame(pca.fit_transform(X))], axis=1)
    # pca2 = PCA(n_components=3, random_state=1337)
    # X = pca.fit_transform(X)
    # pca2 = KernelPCA(n_components=4, kernel='rbf', n_jobs=-1, random_state=1337)
    # pca2.fit(X.iloc[:10000])
    # Xs = []
    # for i in tqdm(range(0, X.shape[0], 1000)):
    #     Xs.append(pd.DataFrame(pca2.transform(X[i:i + 1000])))
    # X = pd.concat(Xs, axis=0)
    X = scaler.fit_transform(X)

    # p = PCA(n_components=20)
    # X = p.fit_transform(X)
    # print(X.head())
    # print(y.head())
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    print(X_train.shape, y_train.shape, X_dev.shape, y_dev.shape)
    return X_train, y_train, X_dev, y_dev  # ,col_names


def rm_features(X_train, y_train, X_dev, y_dev, col_names):
    results = {}
    for col in range(X_train.shape[1]):
        print(col_names[col])
        X_train_t = X_train[:, [c for c in range(X_train.shape[1]) if col != c]]
        X_dev_t = X_dev[:, [c for c in range(X_dev.shape[1]) if col != c]]
        print(X_train_t.shape, X_dev_t.shape)
        clf, time_to_train = train(X_train_t, y_train)
        accuracy, train_accuracy = metrics(clf, X_dev_t, y_dev, X_train_t, y_train)
        results[col_names[col]] = {'train': train_accuracy, 'validation': accuracy}
        print(results[col_names[col]])
    return results


def train(X_train, y_train):
    t0 = time.time()
    clf = RandomForestClassifier(random_state=RANDOM_SEED, verbose=1, max_depth=-1, n_jobs=-1, n_estimators=1000, #max_depth=24
                                 # max_features=2,
                                 criterion='gini')
    clf.fit(X_train, y_train)
    time_to_train = time.time() - t0
    return clf, time_to_train


def predict(clf, X_test):
    predictions = clf.predict(X_test)
    return pd.DataFrame(predictions)


def metrics(clf, X_dev, y_dev, X_train, y_train):
    y_pred = clf.predict(X_dev)

    # conf_mat = confusion_matrix(y_dev, y_pred)
    # roc_auc = roc_auc_score(y_dev, clf.predict_proba(X_dev)[:, 1])
    # f1 = f1_score(y_dev, y_pred)
    accuracy = accuracy_score(y_dev, y_pred)
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    print(clf.feature_importances_)
    # return conf_mat, roc_auc, f1, accuracy, train_accuracy
    return accuracy, train_accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'vectorized_test_data_file', help='file containing vectorized test data')

    parser.add_argument(
        'rf_model_file', help='file containing trained rf model')
    # parser.add_argument(
    #     'rf_model_imputer_file', help='file to contain trained rf model imputer')
    # parser.add_argument(
    #     'rf_model_scaler_file', help='file to contain trained rf model scaler')
    parser.add_argument(
        'rf_model_predictions_file', help='file to contain test predictions')
    args = parser.parse_args()
    # print("Random Forest: Splitting data...")
    # X_train, y_train, X_dev, y_dev = split(args.vectorized_training_data_file, args.labels)
    print("Random Forest: Loading data and model...")
    #imputer_ = pickle.load(open(args.rf_model_imputer_file, 'rb'))
    #scaler = pickle.load(open(args.rf_model_scaler_file, 'rb'))
    X_test = load(args.vectorized_test_data_file)#, imputer_, scaler)
    clf = pickle.load(open(args.rf_model_file, 'rb'))

    # clf, time_to_train = train(X_train, y_train)
    # results = rm_features(X_train, y_train, X_dev, y_dev,col_names)
    print("Random Forest: Getting predictions")
    predictions = predict(clf, X_test)  # X_dev, y_dev, X_train, y_train)

    print("Random Forest: Writing results...")
    predictions.to_csv(args.rf_model_predictions_file)
    # pickle.dump(clf, open(args.rf_model_output_file, 'wb'))
    # metrics_dict = {'accuracy': accuracy, 'time_to_train':time_to_train,#'roc_auc': roc_auc, 'f1': f1, 'time_to_train': time_to_train,
    #                 'train_accuracy': train_accuracy}
    # print(metrics_dict)
    # with open(args.rf_metrics_file, 'w') as metrics_file:
    #     json.dump(metrics_dict, metrics_file)
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
