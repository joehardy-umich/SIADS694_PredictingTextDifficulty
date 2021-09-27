import reference_file_helpers as ref_helpers

def tokenize_doc(doc):
    # just for standalone processing
    # print(doc)
    return [w.lower() for w in word_tokenize(doc) if w not in ignore_tokens]

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
            aoa_stats['mean_age'],
            aoa_stats['median_age'],
            uncommon_words_score,
            avg_concreteness_score,
            unknown_words_score
        ]


def get_data(input_data, aoa, concreteness, common_words):

    df = pd.read_csv(input_data)
    tokenized_sentences = [tokenize_doc(doc) for doc in
                           tqdm(df['original_text'])]

    sg = ScoreGetter(aoa, common_words, concreteness)
    pool = Pool(os.cpu_count() - 1)
    scores = list(tqdm(pool.imap(sg.get_scores, tokenized_sentences, chunksize=128), total=df.shape[0]))
    pool.close()
    pool.join()
    #scores = list(tqdm(map(sg.get_scores, tokenized_sentences), total=df.shape[0]))

    score_df = pd.DataFrame(scores, columns=(
        'Avg_AoA_Score', 'Median_AoA_Score', 'Uncommon_Words_Score', 'Avg_Concreteness_Score', 'Unknown_Words_Score'))

    print(score_df)
    return tokenized_sentences, score_df, df['label']

def do_main_loop():
    parser = argparse.ArgumentParser()

    # INPUTS
    parser.add_argument('input_data', help='file containing input training data')

    parser.add_argument('aoa_data', help='file containing age of acquisition by word data')
    parser.add_argument('concreteness_data', help='file containing concreteness ratings by word data')
    parser.add_argument('dalechall_data', help='file containing dale chall common word list data')

    # OUTPUTS
    parser.add_argument('data_output', help='file to contain output feature representation')
    parser.add_argument('tokenized_sentence_output', help='file to contain output tokenized_sentences')
    parser.add_argument('data_labels', help='file to contain output labels')

    # METRICS
    # N/A

    # MAIN---
    args = parser.parse_args()

    aoa = ref_helpers.get_aoa_dataset(args.aoa_data)
    concreteness = ref_helpers.get_concreteness_dataset(args.concreteness_data)
    common_words = ref_helpers.get_common_words_dataset(
        args.dalechall_data)

    tokenized_sentences, scores, labels = get_data(args.input_data, aoa, concreteness, common_words)

    scores.to_pickle(args.data_output)
    pickle.dump(tokenized_sentences, open(args.tokenized_sentence_output, 'wb'))
    labels.to_pickle(args.data_labels)

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from multiprocessing import Pool
    import os
    import pickle
    from nltk import word_tokenize


    import argparse

    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`'] + list(map(str, range(100)))
    do_main_loop()

