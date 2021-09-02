import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from multiprocessing import Pool
import os
import reference_file_helpers as ref_helpers

RANDOM_SEED = 1337

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

samples = 1000000


def vectorize(original_df):
    df = pd.read_csv(
        original_df
    ).sample(frac=1, random_state=RANDOM_SEED)
    print(df.head())
    tfidf = TfidfVectorizer(stop_words='english', min_df=50, max_features=10000)  # 55,000
    vectorized_df = pd.DataFrame.sparse.from_spmatrix(tfidf.fit_transform(df['original_text']))
    print("Output vectorized dataframe shape: ", str(vectorized_df.shape))
    return vectorized_df, df['label'], tfidf


class LemmaTokenizer:
    # object that TfIdfVectorizer can call
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`'] + list(map(str, range(100)))

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`'] + list(map(str, range(100)))


def tokenize_doc(doc):
    # just for standalone processing
    # print(doc)
    return [w.lower() for w in word_tokenize(doc) if w not in ignore_tokens]


lemma_stops = set(LemmaTokenizer()(' '.join(stop_words)))  # stopwords in the lemmatized version
print(lemma_stops)


def vectorize_lemmatize(original_df, aoa_fp, common_words_fp, concreteness_fp):
    df = pd.read_csv(
        original_df
    ).sample(frac=1, random_state=RANDOM_SEED)
    print(df.head())

    lt = LemmaTokenizer()
    # word needs to show up at least 50 times across the corpus and should not be in more than half the docs
    # tfidf = TfidfVectorizer(stop_words=lemma_stops, tokenizer=lt, min_df=50, max_df=0.5, max_features=10000)  # 55,000
    tfidf = TfidfVectorizer(stop_words=lemma_stops, tokenizer=lt, min_df=10,
                            # , max_df = 0.5 if samples>=10000 else 1.,
                            max_features=10000)

    # standalone_tokenizer = tfidf.build_tokenizer()
    print("Score sentences being tokenized..")
    tokenized_sentences = [tokenize_doc(doc) for doc in
                           tqdm(df.iloc[:samples, 0])]  # [sentence for sentence in df.iloc[:, 0]])))
    # aoa = ref_helpers.get_aoa_dataset(aoa_fp)  # r"../data/reference/AoA_51715_words.csv")
    # common_words = ref_helpers.get_common_words_dataset(common_words_fp)  # r"../data/reference/dale_chall.txt")
    # concreteness = ref_helpers.get_concreteness_dataset(concreteness_fp)
    # for sentence in tokenized_sentences:
    #     # avg_aoa_score = ref_helpers.get_average_age_of_acquisition(sentence, aoa)
    #     # med_aoa_score = ref_helpers.get_median_age_of_acquisition(sentence, aoa)
    #     aoa_stats = ref_helpers.get_age_of_acquisition_stats(sentence, aoa)
    #     common_words_score = ref_helpers.get_count_of_common_words(sentence, common_words)
    #     avg_concreteness_score = ref_helpers.get_average_concreteness_rating(sentence, concreteness)
    # return
    sparse_df = tfidf.fit_transform(df.iloc[:samples]['original_text'])
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(sparse_df)
    print(tfidf.vocabulary_, len(tfidf.vocabulary_))
    vectorized_df = add_aggregated_scores(
        tfidf_df,
        tokenized_sentences, aoa_fp, common_words_fp, concreteness_fp)

    print("Output vectorized dataframe shape: ", str(vectorized_df.shape))
    return vectorized_df, df.iloc[:samples]['label'], tfidf, lt


# def get_scores(sentence, aoa, common_words, concreteness):
#     aoa_stats = ref_helpers.get_age_of_acquisition_stats(sentence, aoa)
#     uncommon_words_score = ref_helpers.get_count_of_uncommon_words(sentence, common_words)
#     avg_concreteness_score = ref_helpers.get_average_concreteness_rating(sentence, concreteness)
#     unknown_words_score = ref_helpers.get_count_of_unknown_words(sentence, concreteness)
#     return [
#         aoa_stats['mean'],
#         aoa_stats['median'],
#         uncommon_words_score,
#         avg_concreteness_score,
#         unknown_words_score
#     ]

class ScoreGetter(object):
    def __init__(self,aoa,common_words,concreteness):
        self.aoa=aoa
        self.common_words=common_words
        self.concreteness=concreteness
    def get_scores(self,sentence):
        aoa_stats = ref_helpers.get_age_of_acquisition_stats(sentence, self.aoa)
        uncommon_words_score = ref_helpers.get_count_of_uncommon_words(sentence, self.common_words)
        avg_concreteness_score = ref_helpers.get_average_concreteness_rating(sentence, self.concreteness)
        unknown_words_score = ref_helpers.get_count_of_unknown_words(sentence, self.concreteness)
        return [
            aoa_stats['mean'],
            aoa_stats['median'],
            uncommon_words_score,
            avg_concreteness_score,
            unknown_words_score
        ]

def add_aggregated_scores(original_df, tokenized_sentences, aoa_fp, common_words_fp, concreteness_fp):
    # take a vectorized dataframe and append scores (given by functions in reference_file_helpers.py helpers section)
    aoa = ref_helpers.get_aoa_dataset(aoa_fp)  # r"../data/reference/AoA_51715_words.csv")
    common_words = ref_helpers.get_common_words_dataset(common_words_fp)  # r"../data/reference/dale_chall.txt")
    concreteness = ref_helpers.get_concreteness_dataset(concreteness_fp)
    # r"../data/reference/Concreteness_ratings_Brysbaert_et_al_BRM.txt")
    # scores = []
    print("Assessing sentence scores..")
    pool = Pool(os.cpu_count())
    sg = ScoreGetter(aoa,common_words,concreteness)
    #f = lambda s: get_scores(s, aoa, common_words, concreteness)
    scores = pool.map(sg.get_scores, tqdm(tokenized_sentences))
    pool.close()
    pool.join()
    print(tokenized_sentences[0], scores[0])
    # for sentence in tqdm(tokenized_sentences):
    #     # avg_aoa_score = ref_helpers.get_average_age_of_acquisition(sentence, aoa)
    #     # med_aoa_score = ref_helpers.get_median_age_of_acquisition(sentence, aoa)
    #     aoa_stats = ref_helpers.get_age_of_acquisition_stats(sentence, aoa)
    #     uncommon_words_score = ref_helpers.get_count_of_uncommon_words(sentence, common_words)
    #     avg_concreteness_score = ref_helpers.get_average_concreteness_rating(sentence, concreteness)
    #     unknown_words_score = ref_helpers.get_count_of_unknown_words(sentence, concreteness)
    #     scores.append([
    #         aoa_stats['mean'],
    #         aoa_stats['median'],
    #         uncommon_words_score,
    #         avg_concreteness_score,
    #         unknown_words_score
    #     ])
    score_df = pd.DataFrame(scores, columns=(
        'Avg_AoA_Score', 'Median_AoA_Score', 'Uncommon_Words_Score', 'Avg_Concreteness_Score', 'Unknown_Words_Score'))

    score_df = (score_df - score_df.min()) / (score_df.max() - score_df.min())
    print(pd.concat([original_df, score_df], axis=1))
    return pd.concat([original_df, score_df], axis=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'training_data_file', help='file containing training data')
    parser.add_argument(
        'aoa_data_file', help='file containing word Age of Acquisition Data')
    parser.add_argument(
        'concreteness_data_file', help='file containing word concreteness score data')
    parser.add_argument(
        'common_words_data_file', help='file containing most common word data')
    parser.add_argument(
        'vectorized_training_data_output_file', help='file to contain vectorized training data')
    parser.add_argument(
        'labels_output_file', help='file to contain labels')
    parser.add_argument(
        'trained_tfidf_output_file', help='file to contain vectorized training data tfidf transformer')
    parser.add_argument(
        'lemmatizer_output_file', help='file to contain lemma tokenizer')
    args = parser.parse_args()

    vectorized, labels, tfidf, lemmatizer = vectorize_lemmatize(args.training_data_file, args.aoa_data_file,
                                                                args.common_words_data_file,
                                                                args.concreteness_data_file)
    vectorized.to_pickle(args.vectorized_training_data_output_file)
    labels.to_pickle(args.labels_output_file)
    pickle.dump(tfidf, open(args.trained_tfidf_output_file, 'wb'))
    pickle.dump(lemmatizer, open(args.lemmatizer_output_file, 'wb'))
