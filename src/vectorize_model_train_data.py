import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

RANDOM_SEED = 1337


def vectorize(original_df):
    df = pd.read_csv(
        original_df
    ).sample(frac=1, random_state=RANDOM_SEED)
    print(df.head())
    tfidf = TfidfVectorizer(stop_words='english', min_df=50, max_features=10000)
    vectorized_df = pd.DataFrame.sparse.from_spmatrix(tfidf.fit_transform(df['original_text']))
    print("Output vectorized dataframe shape: ", str(vectorized_df.shape))
    return vectorized_df, df['label'], tfidf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'training_data_file', help='file containing training data')
    parser.add_argument(
        'vectorized_training_data_output_file', help='file to contain vectorized training data')
    parser.add_argument(
        'labels_output_file', help='file to contain labels')
    parser.add_argument(
        'trained_tfidf_output_file', help='file to contain vectorized training data tfidf transformer')
    args = parser.parse_args()

    vectorized, labels, tfidf = vectorize(args.training_data_file)
    vectorized.to_pickle(args.vectorized_training_data_output_file)
    labels.to_pickle(args.labels_output_file)
    pickle.dump(tfidf, open(args.trained_tfidf_output_file, 'wb'))
