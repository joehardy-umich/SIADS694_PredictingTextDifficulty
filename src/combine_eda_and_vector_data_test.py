import os

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


def combine(eda, vector):
    eda_df = pd.read_pickle(eda)
    #eda_df[pd.isnull(eda_df)] = 0.

    vector_df = get_summed_cols(pd.read_pickle(vector).iloc[:, :-5],'tfidf')

    print(eda_df.shape, vector_df.shape)

    return pd.concat([eda_df, vector_df], axis=1)


def get_summed_cols(X,name):
    print(X.shape)
    X = X.astype(pd.SparseDtype('float', 0.))
    col_sum = None
    for col in tqdm(X.columns):
        if str(X[col].dtype) == 'Sparse[float64, 0]' or str(X[col].dtype) == 'Sparse[float64, 0.0]':
            X[col] = X[col].sparse.to_dense()
            X.loc[X[col] < 0, col] = 0.
            X.loc[pd.isnull(X[col]), col] = 0.
            if col not in limit_to:
                if col_sum is not None:
                    col_sum += X[col]
                else:
                    col_sum = X[col]
                X = X.drop(columns=col)
            # X[col] = pd.arrays.SparseArray(X[col])
        else:
            if col not in limit_to:
                if col_sum is not None:
                    col_sum += X[col]
                else:
                    col_sum = X[col]
                X = X.drop(columns=col)

    X['sentence_sum_'+name] = col_sum
    return X


if __name__ == "__main__":
    import pandas as pd
    import argparse
    import pickle
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'eda_training_data_file', help='file containing eda training data')
    parser.add_argument(
        'vector_training_data_file', help='file containing vector training data')
    parser.add_argument(
        'data_output_file', help='file to contain combined baseline model')
    args = parser.parse_args()

    print("Combining data...")
    combined = combine(args.eda_training_data_file, args.vector_training_data_file)

    pickle.dump(combined, open(args.data_output_file, 'wb'))
