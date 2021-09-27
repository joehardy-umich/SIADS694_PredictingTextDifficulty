import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.impute import SimpleImputer as imputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import csv

def build_dfs(vectorized_train):
    X = pd.read_pickle(vectorized_train)
    X[pd.isnull(X)] = 0.
    X = X.astype(pd.SparseDtype('float', 0.))
    X['sum_all'] = X['sum_1'] + X['sum_0'] + X['sum_none']
    X['sum_ratio'] = X['sum_1'] / (X['sum_0'] + X['sum_none'] + 1)
    X['sum_diff'] = X['sum_1']-X['sum_0']
    X=X.sparse.to_dense()
    cols = X.columns
    # imputer_ = imputer(missing_values=-1,strategy='mean',add_indicator=False)
    # X = imputer_.fit_transform(X)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # X = X.astype(pd.SparseDtype('float', 0.))
    print(type(X))
    X = pd.DataFrame(X,columns=cols)
    X_limited = pd.DataFrame(X,columns=cols).loc[:,['sum_diff', 'mean_age', 'mean_concreteness', 'mean_freq_pm', 'min_perc_known']]
    print("Kmeans: Step 1 - file read in", X.shape)
    return X, X_limited

def kselector(X_limited):
    k_values = range(2,41)
    inertias_list = []
    dbs_list = []
    chs_list = []
    silhouette_list = []
    for i in k_values:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42).fit(X_limited)
        labels = kmeans.labels_
        inertias_list.append(kmeans.inertia_)
        dbs = davies_bouldin_score(X_limited, labels)
        chs = calinski_harabasz_score(X_limited, labels)
        # silhouettescore = silhouette_score(X_limited, labels, metric='euclidean')
        dbs_list.append(dbs)
        chs_list.append(chs)
        # silhouette_list.append(silhouettescore)
        print(i)
    kmeans_df = pd.DataFrame({'k_value': k_values, 'intertia': inertias_list, 'davies_bouldin_score': dbs_list, 'calinski_harabasz_score': chs_list})
    return kmeans_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'vectorized_training_data_file', help='file containing vectorized training data')
    parser.add_argument(
        'kmeans_output_file', help='kmeans_output_file')
    args = parser.parse_args()
    X, X_limited = build_dfs(args.vectorized_training_data_file)
    kmeans_df = kselector(X_limited)
    kmeans_df.to_csv(args.kmeans_output_file, index=False)