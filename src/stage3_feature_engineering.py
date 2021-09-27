def get_transformed_data(df_path, aoa, concreteness, common_words,
                         count_threshold=100, label_perc_diff_threshold=0.2):
    nltk.download('averaged_perceptron_tagger')
    tqdm.pandas()
    df = pd.read_csv(df_path)
    print(df.columns)
    labels = df['label']
    tokenized_text = df['original_text'].str.lower().progress_apply(word_tokenize)
    word_label_dict = {}
    tag_label_dict = {}
    pair_label_dict = {}
    tagged_sentences = []
    tagged_sentence_pairs = []

    print("Tagging sentences (1 of 5)")
    for sentence, label in tqdm(zip(tokenized_text, labels), total=df.shape[0]):
        tagged_sentence = pos_tag(sentence)
        tagged_sentences.append(tagged_sentence)
        for word, tag in tagged_sentence:
            word = word.lower()
            if word not in word_label_dict:
                word_label_dict[word] = {}
                word_label_dict[word][label] = 1
            else:
                if label not in word_label_dict[word]:
                    word_label_dict[word][label] = 1
                else:
                    word_label_dict[word][label] += 1
            if tag not in tag_label_dict:
                tag_label_dict[tag] = {}
                tag_label_dict[tag][label] = 1
            else:
                if label not in tag_label_dict[tag]:
                    tag_label_dict[tag][label] = 1
                else:
                    tag_label_dict[tag][label] += 1
        if len(tagged_sentence) > 1:
            tag_pairs = [[tag[1] for tag in tagged_sentence[i:i + 2]] for i in range(0, len(tagged_sentence) - 2)]
            tagged_sentence_pairs.append(tag_pairs)
            for pair in tag_pairs:
                pair = str(pair)
                if pair not in pair_label_dict:
                    pair_label_dict[pair] = {}
                    pair_label_dict[pair][label] = 1
                else:
                    if label not in pair_label_dict[pair]:
                        pair_label_dict[pair][label] = 1
                    else:
                        pair_label_dict[pair][label] += 1
        else:
            tagged_sentence_pairs.append([])
    print("Constructing threshold and frequencies for words (2 of 5)")
    for word in tqdm(word_label_dict):
        sum_labels = sum(word_label_dict[word].values())
        word_label_dict[word]['total'] = sum_labels
        for label in list(word_label_dict[word].keys()):
            word_label_dict[word][str(label) + ":perc"] = word_label_dict[word][label] / sum_labels
        if '0:perc' not in word_label_dict[word]:
            word_label_dict[word]['0:perc'] = 0.
        if '1:perc' not in word_label_dict[word]:
            word_label_dict[word]['1:perc'] = 0.

    # word_df = pd.DataFrame.from_records(word_label_dict)
    # word_df.T.to_csv(word_eda_path)
    #
    # tag_df = pd.DataFrame.from_records(tag_label_dict)
    # tag_df.T.to_csv(tag_eda_path)
    #
    # pair_df = pd.DataFrame.from_records(pair_label_dict)
    # pair_df.T.to_csv(pair_eda_path)

    print("Constructing threshold and frequencies for tags (3 of 5)")
    for tag in tqdm(tag_label_dict):
        sum_labels = sum(tag_label_dict[tag].values())
        tag_label_dict[tag]['total'] = sum_labels
        for label in list(tag_label_dict[tag].keys()):
            tag_label_dict[tag][str(label) + ":perc"] = tag_label_dict[tag][label] / sum_labels
        if '0:perc' not in tag_label_dict[tag]:
            tag_label_dict[tag]['0:perc'] = 0.
        if '1:perc' not in tag_label_dict[tag]:
            tag_label_dict[tag]['1:perc'] = 0.

    print("Constructing threshold and frequencies for tag pairs (4 of 5)")
    for pair in tqdm(pair_label_dict):
        sum_labels = sum(pair_label_dict[pair].values())
        pair_label_dict[pair]['total'] = sum_labels
        for label in list(pair_label_dict[pair].keys()):
            pair_label_dict[pair][str(label) + ":perc"] = pair_label_dict[pair][label] / sum_labels
        if '0:perc' not in pair_label_dict[pair]:
            pair_label_dict[pair]['0:perc'] = 0.
        if '1:perc' not in pair_label_dict[pair]:
            pair_label_dict[pair]['1:perc'] = 0.

    print("Constructing data records (5 of 5)")
    data_records = []
    for tagged_sentence, tagged_pairs in tqdm(zip(tagged_sentences, tagged_sentence_pairs), total=df.shape[0]):
        data_record = {'sum_1': 0, 'sum_0': 0, 'sum_none': 0}
        words, tags = zip(*tagged_sentence)

        pairs = [str(pair) for pair in tagged_pairs]
        for word in words:
            if word_label_dict[word]['0:perc'] > word_label_dict[word]['1:perc']:
                data_record['sum_0'] += 1
            elif word_label_dict[word]['1:perc'] > word_label_dict[word]['0:perc']:
                data_record['sum_1'] += 1
            else:
                data_record['sum_none'] += 1
        for tag in tags:
            if tag_label_dict[tag]['0:perc'] > tag_label_dict[tag]['1:perc']:
                data_record['sum_0'] += 1
            elif tag_label_dict[tag]['1:perc'] > tag_label_dict[tag]['0:perc']:
                data_record['sum_1'] += 1
            else:
                data_record['sum_none'] += 1

        for pair in pairs:
            if pair_label_dict[pair]['0:perc'] > pair_label_dict[pair]['1:perc']:
                data_record['sum_0'] += 1
            elif pair_label_dict[pair]['1:perc'] > pair_label_dict[pair]['0:perc']:
                data_record['sum_1'] += 1
            else:
                data_record['sum_none'] += 1

        # Get Scores
        aoa_dict = ref_helpers.get_age_of_acquisition_stats(words, aoa)
        concreteness_dict = ref_helpers.get_concreteness_stats(words, concreteness)
        dale_chall_dict = ref_helpers.get_dale_chall_stats(words, common_words)
        word_stats_dict = ref_helpers.get_word_stats(words)

        data_record.update(aoa_dict)
        data_record.update(concreteness_dict)
        data_record.update(dale_chall_dict)
        data_record.update(word_stats_dict)

        data_records.append(data_record)
    rdf = pd.DataFrame(data_records)
    rdf[pd.isnull(df)] = 0.
    rdf['sum_all'] = rdf['sum_1'] + rdf['sum_0'] + rdf['sum_none']
    rdf['sum_ratio'] = rdf['sum_1'] / (rdf['sum_0'] + rdf['sum_none'] + 1)
    rdf['sum_diff'] = rdf['sum_1'] - rdf['sum_0']
    return rdf, df['label'], tagged_sentences


if __name__ == "__main__":
    import pandas as pd
    import nltk
    from nltk import word_tokenize, pos_tag
    from tqdm import tqdm
    import argparse
    import reference_file_helpers as ref_helpers
    import pickle

    parser = argparse.ArgumentParser()

    # INPUTS
    parser.add_argument(
        'training_data_file', help='file containing training data')
    parser.add_argument(
        'aoa_data_file', help='file containing word Age of Acquisition Data')
    parser.add_argument(
        'concreteness_data_file', help='file containing word concreteness score data')
    parser.add_argument(
        'common_words_data_file', help='file containing most common word data')

    # OUTPUTS
    parser.add_argument(
        'vectorized_training_data_output_file', help='file to contain vectorized training data')
    parser.add_argument(
        'labels_output_file', help='file to contain labels')
    parser.add_argument(
        'tagged_sentences_output_file', help='file to contain tagged sentences')

    # METRICS
    # N/A

    # MAIN---
    args = parser.parse_args()

    aoa = ref_helpers.get_aoa_dataset(args.aoa_data_file)
    concreteness = ref_helpers.get_concreteness_dataset(args.concreteness_data_file)
    common_words = ref_helpers.get_common_words_dataset(args.common_words_data_file)

    transformed_data, labels, tagged_sentences = get_transformed_data(args.training_data_file, aoa, concreteness,
                                                                      common_words)

    transformed_data.to_pickle(args.vectorized_training_data_output_file)
    labels.to_pickle(args.labels_output_file)
    pickle.dump(tagged_sentences, open(args.tagged_sentences_output_file, 'wb'))
