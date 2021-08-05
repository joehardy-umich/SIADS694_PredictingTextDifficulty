import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def vectorize(original_df):
    df = pd.read_csv(
        original_df
    )
    tfidf = TfidfVectorizer(stop_words='english', min_df=5)
    vectorized_df = pd.DataFrame(tfidf.fit_transform(df['original_text']))

    return vectorized_df, tfidf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'training_data_file', help='file containing training data')
    parser.add_argument(
        'vectorized_training_data_output_file', help='file to contain vectorized training data')
    parser.add_argument(
        'trained_tfidf_output_file', help='file to contain vectorized training data tfidf transformer')
    args = parser.parse_args()

    vectorized, tfidf = vectorize(args.training_data_file)
    vectorized.to_csv(args.vectorized_training_data_output_file, index=False)
    pickle.dump(tfidf, open(args.trained_tfidf_output_file, 'wb'))
