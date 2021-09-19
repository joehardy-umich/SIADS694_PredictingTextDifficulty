import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
import pickle
import time

RANDOM_SEED = 1337

def build_dfs(vectorized_train):
    X = pd.read_pickle(vectorized_train).astype(pd.SparseDtype('float', 0.))
    print("Kmeans: Step 1 - file read in", X.shape)
    return X

def train(X):
    t0 = time.time()
    kmeans = KMeans(n_clusters=11, init='k-means++', max_iter=50, n_init=1, random_state=42).fit(X)
    time_to_train = time.time() - t0
    print("Kmeans: Step 2 - model trained")
    return kmeans, time_to_train

def kmeans_preds(X, kmeans):
    n = 37888  #chunk row size
    list_df = [X[i:i+n] for i in range(0,X.shape[0],n)] #producing chunks
    final_predictions = np.array([])
    for i in range(0,11):
        predict=kmeans.predict(list_df[i])
        final_predictions = np.append(final_predictions,predict)
        print("Kmeans: Step 3 - predictions, part", i, "of 10")
    X['unsupervised_learning_clusterID'] = final_predictions.tolist()
    return X

def metrics(X_sampled, kmeans):
    labels = kmeans.labels_
    dbs = davies_bouldin_score(X_sampled, labels)
    chs = calinski_harabasz_score(X_sampled, labels)
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
    X = build_dfs(args.vectorized_training_data_file)
    print("Kmeans: Step 2 - training model...")
    kmeans, time_to_train = train(X)
    print("Kmeans: Step 3 - predicting and outputting new training data")
    X = kmeans_preds(X, kmeans)
    X.to_pickle(args.output_file_with_new_kmeanscolumn)
    print("Kmeans: Step 4 - getting metrics and writing results...")
    dbs, chs = metrics(X, kmeans)
    pickle.dump(kmeans, open(args.model_kmeans_output_file, 'wb'))
    metrics_dict = {'davies_bouldin_score': dbs, 'calinski_harabasz_score': chs, 'time_to_train': time_to_train}
    with open(args.model_kmeans_metrics_file, 'w') as metrics_file:
        json.dump(metrics_dict, metrics_file)
    print("Kmeans: Pipeline complete")