import pandas as pd
#from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def vectorize(original_df,tfidf_file):
    df = pd.read_csv(
        original_df
    )
    #tfidf = TfidfVectorizer(stop_words='english', min_df=5)
    tfidf = pickle.load(open(tfidf_file,'rb'))
    vectorized_df = pd.DataFrame.sparse.from_spmatrix(tfidf.transform(df['original_text']))
    return vectorized_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test_data_file', help='file containing test data')
    parser.add_argument(
        'trained_tfidf_input_file', help='file to contain vectorized training data tfidf transformer')
    parser.add_argument(
        'vectorized_test_data_output_file', help='file to contain vectorized test data')

    args = parser.parse_args()

    vectorized = vectorize(args.test_data_file,args.trained_tfidf_input_file)
    vectorized.to_pickle(args.vectorized_test_data_output_file, index=False)
