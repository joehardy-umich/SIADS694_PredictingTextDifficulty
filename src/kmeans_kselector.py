import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

def kselector(input):
    k_lst = []
    k_values = (2,3,4,5,6,7,8,9,10,11,12,13,14,15)
    for i in k_values:
        X = pd.read_pickle(input).astype(pd.SparseDtype('float', 0.))
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=1, random_state=42).fit(X)
        labels = kmeans.labels_
        dbs = davies_bouldin_score(X, labels)
        chs = calinski_harabasz_score(X, labels)
        k_lst.append([i, dbs, chs])
    return print(k_lst)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'vectorized_training_data_file', help='file containing vectorized training data')
    args = parser.parse_args()
    k_lst = kselector(args.vectorized_training_data_file)