import reference_file_helpers as ref_helpers

RANDOM_SEED = 1337

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


samples = 1000000


def vectorize(original_df,tfidf):
    df = pd.read_csv(
        original_df
    )#.sample(frac=1, random_state=RANDOM_SEED)
    print(df.head())
    #tfidf = TfidfVectorizer(stop_words='english', min_df=50, max_features=10000)  # 55,000
    vectorized_df = pd.DataFrame.sparse.from_spmatrix(tfidf.fit_transform(df['original_text']))
    print("Output vectorized dataframe shape: ", str(vectorized_df.shape))
    return vectorized_df, tfidf


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


def get_lemma_stops():
    stop_words = set(stopwords.words('english'))
    lemma_stops = set(LemmaTokenizer()(' '.join(stop_words)))  # stopwords in the lemmatized version
    return lemma_stops


def get_stops():
    return set(stopwords.words('english'))


def vectorize_lemmatize(original_df):
    df = pd.read_csv(
        original_df
    )[['original_text']]
    print(df.head())
    # lemma_stops = get_lemma_stops()
    # lt = LemmaTokenizer()
    # word needs to show up at least 50 times across the corpus and should not be in more than half the docs
    # tfidf = TfidfVectorizer(stop_words=lemma_stops, tokenizer=lt, min_df=50, max_df=0.5, max_features=10000)  # 55,000
    # tfidf = TfidfVectorizer(stop_words=get_lemma_stops(), tokenizer=lt, min_df=10,
    #                         max_df=0.5 if samples >= 10000 else 1.,
    #                         max_features=10000)

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
    return df, tokenized_sentences


def tfidf_and_combine(df, tfidf):
    sparse_df = tfidf.transform(df.iloc[:samples]['original_text'])
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(sparse_df)
    print(tfidf.vocabulary_, len(tfidf.vocabulary_))
    # vectorized_df = add_aggregated_scores(
    #     tfidf_df,
    #     tokenized_sentences, aoa_fp, common_words_fp, concreteness_fp)

    # vectorized_df = pd.concat([tfidf_df, score_df], axis=1)
    print("Output vectorized dataframe shape: ", str(tfidf_df.shape))
    return tfidf_df


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
    def __init__(self, aoa, common_words, concreteness):
        self.aoa = aoa
        self.common_words = common_words
        self.concreteness = concreteness

    def get_scores(self, sentence):
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
    # pool = Pool(os.cpu_count()-1)
    sg = ScoreGetter(aoa, common_words, concreteness)
    # f = lambda s: get_scores(s, aoa, common_words, concreteness)
    # scores = pool.map(sg.get_scores, tqdm(tokenized_sentences))
    # pool.close()
    # pool.join()
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
    return score_df, score_df.min(), score_df.max()
    # print(pd.concat([original_df, score_df], axis=1))
    # return pd.concat([original_df, score_df], axis=1)


def do():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test_data_file', help='file containing test data')
    parser.add_argument(
        'vectorized_test_data_output_file', help='file to contain vectorized test data')
    parser.add_argument(
        'trained_tfidf_file', help='file containing pre-fit vectorized test data tfidf transformer')
    parser.add_argument(
        'tokenized_sentences_output_file', help='file to contain non-lemmatized tokenized sentences')
    args = parser.parse_args()

    # vectorized, labels, tfidf, lemmatizer = vectorize_lemmatize(args.test_data_file, args.aoa_data_file,
    #                                                             args.common_words_data_file,
    #                                                             args.concreteness_data_file)
    tfidf = pickle.load(open(args.trained_tfidf_file,'rb'))
    df, tokenized_sentences = vectorize_lemmatize(args.test_data_file)
    pickle.dump(tokenized_sentences, open(args.tokenized_sentences_output_file, 'wb'))
    # aoa = ref_helpers.get_aoa_dataset(args.aoa_data_file)  # r"../data/reference/AoA_51715_words.csv")
    # common_words = ref_helpers.get_common_words_dataset(
    #     args.common_words_data_file)  # r"../data/reference/dale_chall.txt")
    # concreteness = ref_helpers.get_concreteness_dataset(args.concreteness_data_file)
    # r"../data/reference/Concreteness_ratings_Brysbaert_et_al_BRM.txt")
    # scores = []
    # print("Assessing sentence scores..")
    # pool = Pool(os.cpu_count() - 1)
    # sg = ScoreGetter(aoa, common_words, concreteness)
    # # f = lambda s: get_scores(s, aoa, common_words, concreteness)
    # # scores = process_map(sg.get_scores,range(len(df)),chunksize=1)
    # scores = list(tqdm(pool.imap(sg.get_scores, tokenized_sentences, chunksize=128), total=df.shape[0]))
    # pool.close()
    # pool.join()
    # print(tokenized_sentences[0], scores[0])
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
    # score_df = pd.DataFrame(scores, columns=(
    #     'Avg_AoA_Score', 'Median_AoA_Score', 'Uncommon_Words_Score', 'Avg_Concreteness_Score', 'Unknown_Words_Score'))
    # 
    # score_df = (score_df - score_df.min()) / (score_df.max() - score_df.min())

    vectorized = tfidf_and_combine(df, tfidf)
    vectorized.to_pickle(args.vectorized_test_data_output_file)


if __name__ == '__main__':
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    import pickle
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import nltk
    from nltk.corpus import stopwords
    from tqdm import tqdm
    from multiprocessing import Pool
    from tqdm.contrib.concurrent import process_map
    import os

    do()
