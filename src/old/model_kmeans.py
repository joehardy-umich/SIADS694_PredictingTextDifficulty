import json
from os import X_OK
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.impute import SimpleImputer as imputer
from sklearn.preprocessing import StandardScaler
import pickle
import time

RANDOM_SEED = 1337

def build_dfs(vectorized_train):
    X = pd.read_pickle(vectorized_train)
    X[pd.isnull(X)] = 0.
    X = X.astype(pd.SparseDtype('float', 0.))
    X['sum_all'] = X['sum_1'] + X['sum_0'] + X['sum_none']
    X['sum_ratio'] = X['sum_1'] / (X['sum_0'] + X['sum_none'] + 1)
    X['sum_diff'] = X['sum_1']-X['sum_0']
    X=X.sparse.to_dense()
    cols = X.columns
    imputer_ = imputer(missing_values=-1,strategy='mean',add_indicator=False)
    X = imputer_.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # X = X.astype(pd.SparseDtype('float', 0.))
    print(type(X))
    X = pd.DataFrame(X,columns=cols)
    X_limited = pd.DataFrame(X,columns=cols).loc[:,['sum_diff', 'mean_age', 'mean_concreteness', 'mean_freq_pm', 'min_perc_known']]
    print("Kmeans: Step 1 - file read in", X.shape)
    return X, X_limited

def train(X_limited):
    t0 = time.time()
    kmeans = KMeans(n_clusters=25, init='k-means++', max_iter=300, n_init=10, random_state=42,verbose=1).fit(X_limited)
    time_to_train = time.time() - t0
    print("Kmeans: Step 2 - model trained")
    return kmeans, time_to_train

def kmeans_preds(X, X_limited, kmeans):
    n = 37888  #chunk row size
    list_df = [X_limited[i:i+n] for i in range(0,X_limited.shape[0],n)] #producing chunks
    print(len(list_df),X_limited.shape[0])
    final_predictions = np.array([])
    for i in range(0,11):
        predict=kmeans.predict(list_df[i])
        final_predictions = np.append(final_predictions,predict)
        print("Kmeans: Step 3 - predictions, part", i, "of 10")
    X['unsupervised_learning_clusterID'] = final_predictions.tolist()
    return X

def metrics(X_limited, kmeans):
    labels = kmeans.labels_
    dbs = davies_bouldin_score(X_limited, labels)
    chs = calinski_harabasz_score(X_limited, labels)
    print("Kmeans: Step 4 - metrics and results written")
    return dbs, chs

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'vectorized_training_data_file', help='file containing vectorized training data')
    parser.add_argument(
        'output_file_with_new_kmeanscolumn', help='file to contain new kmeans clustered column')
    parser.add_argument(
        'model_kmeans_output_file', help='file to contain trained kmeans model')
    parser.add_argument(
        'model_kmeans_metrics_file', help='file to contain trained kmeans model metrics')
    args = parser.parse_args()
    print("Kmeans: Step 1 - loading data, and creating sampled df for training...")
    X, X_limited = build_dfs(args.vectorized_training_data_file)
    print("Kmeans: Step 2 - training model...")
    kmeans, time_to_train = train(X_limited)
    print("Kmeans: Step 3 - predicting and outputting new training data")
    X = kmeans_preds(X, X_limited, kmeans)
    X.to_pickle(args.output_file_with_new_kmeanscolumn)
    print("Kmeans: Step 4 - getting metrics and writing results...")
    dbs, chs = metrics(X_limited, kmeans)
    pickle.dump(kmeans, open(args.model_kmeans_output_file, 'wb'))
    metrics_dict = {'davies_bouldin_score': dbs, 'calinski_harabasz_score': chs, 'time_to_train': time_to_train}
    with open(args.model_kmeans_metrics_file, 'w') as metrics_file:
        json.dump(metrics_dict, metrics_file)
    print("Kmeans: Pipeline complete")