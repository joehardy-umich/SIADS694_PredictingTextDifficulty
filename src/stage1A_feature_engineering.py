from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def get_lemma_stops():
    stop_words = set(stopwords.words('english'))
    lemma_stops = set(LemmaTokenizer()(' '.join(stop_words)))  # stopwords in the lemmatized version
    return lemma_stops


class LemmaTokenizer:
    # object that TfIdfVectorizer can call
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`'] + list(map(str, range(100)))

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


lt = LemmaTokenizer()
tfidf = TfidfVectorizer(stop_words=get_lemma_stops(), tokenizer=lt, lowercase=True,
                        min_df=10, max_df=0.5,
                        max_features=10000)


def get_data(input_data):
    df = pd.read_csv(input_data)

    sparse_df = tfidf.fit_transform(df['original_text'])
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(sparse_df)
    return tfidf_df, df['label']


if __name__ == "__main__":
    import pandas as pd


    import argparse

    # INPUTS
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', help='file containing input training data')

    # OUTPUTS
    parser.add_argument('data_output', help='file to contain output feature representation')
    parser.add_argument('data_labels', help='file to contain output labels')

    # METRICS
    # N/A

    # MAIN ---
    args = parser.parse_args()

    data, labels = get_data(args.input_data)

    data.to_pickle(args.data_output)
    labels.to_pickle(args.data_labels)
