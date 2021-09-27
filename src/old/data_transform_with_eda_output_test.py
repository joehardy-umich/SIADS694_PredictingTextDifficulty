def get_transformed_data(df_path, aoa, concreteness, common_words, word_eda_path, tag_eda_path,
                         pair_eda_path):  # , count_threshold=1000, label_perc_diff_threshold=0.2):
    nltk.download('averaged_perceptron_tagger')
    tqdm.pandas()
    df = pd.read_csv(df_path)
    print(df.columns)
    tokenized_text = df['original_text'].str.lower().progress_apply(word_tokenize)
    word_label_dict = {}
    tag_label_dict = {}
    pair_label_dict = {}
    tagged_sentences = []
    tagged_sentence_pairs = []
    for sentence in tqdm(tokenized_text, total=df.shape[0]):
        tagged_sentence = pos_tag(sentence)
        tagged_sentences.append(tagged_sentence)
        # for word, tag in tagged_sentence:
        #     word = word.lower()
        #     if word not in word_label_dict:
        #         word_label_dict[word] = {}
        #         word_label_dict[word][label] = 1
        #     else:
        #         if label not in word_label_dict[word]:
        #             word_label_dict[word][label] = 1
        #         else:
        #             word_label_dict[word][label] += 1
        #     if tag not in tag_label_dict:
        #         tag_label_dict[tag] = {}
        #         tag_label_dict[tag][label] = 1
        #     else:
        #         if label not in tag_label_dict[tag]:
        #             tag_label_dict[tag][label] = 1
        #         else:
        #             tag_label_dict[tag][label] += 1
        if len(tagged_sentence) > 1:
            tag_pairs = [[tag[1] for tag in tagged_sentence[i:i + 2]] for i in range(0, len(tagged_sentence) - 2)]
            tagged_sentence_pairs.append(tag_pairs)
            # for pair in tag_pairs:
            #     pair = str(pair)
            #     if pair not in pair_label_dict:
            #         pair_label_dict[pair] = {}
            #         pair_label_dict[pair][label] = 1
            #     else:
            #         if label not in pair_label_dict[pair]:
            #             pair_label_dict[pair][label] = 1
            #         else:
            #             pair_label_dict[pair][label] += 1
        else:
            tagged_sentence_pairs.append([])

    word_label_dict = pd.read_csv(word_eda_path, index_col=0).rename(columns={'0': 0, '1': 1}).reset_index().drop_duplicates('index').set_index(
        'index').to_dict(orient='index')

    tag_label_dict = pd.read_csv(tag_eda_path, index_col=0).rename(columns={'0': 0, '1': 1}).reset_index().drop_duplicates('index').set_index(
        'index').to_dict(orient='index')

    pair_label_dict = pd.read_csv(pair_eda_path, index_col=0).rename(columns={'0': 0, '1': 1}).reset_index().drop_duplicates('index').set_index(
        'index').to_dict(orient='index')

    for word in tqdm(word_label_dict):
        sum_labels = sum([v for k, v in word_label_dict[word].items() if k in [0, 1]])
        word_label_dict[word]['total'] = sum_labels
        for label in list(word_label_dict[word].keys()):
            word_label_dict[word][str(label) + ":perc"] = word_label_dict[word][label] / sum_labels
        if '0:perc' not in word_label_dict[word]:
            word_label_dict[word]['0:perc'] = 0.
        if '1:perc' not in word_label_dict[word]:
            word_label_dict[word]['1:perc'] = 0.

    for tag in tqdm(tag_label_dict):
        sum_labels = sum([v for k, v in tag_label_dict[tag].items() if k in [0, 1]])
        tag_label_dict[tag]['total'] = sum_labels
        for label in list(tag_label_dict[tag].keys()):
            tag_label_dict[tag][str(label) + ":perc"] = tag_label_dict[tag][label] / sum_labels
        if '0:perc' not in tag_label_dict[tag]:
            tag_label_dict[tag]['0:perc'] = 0.
        if '1:perc' not in tag_label_dict[tag]:
            tag_label_dict[tag]['1:perc'] = 0.

    for pair in tqdm(pair_label_dict):
        sum_labels = sum([v for k, v in pair_label_dict[pair].items() if k in [0, 1]])
        pair_label_dict[pair]['total'] = sum_labels
        for label in list(pair_label_dict[pair].keys()):
            pair_label_dict[pair][str(label) + ":perc"] = pair_label_dict[pair][label] / sum_labels
        if '0:perc' not in pair_label_dict[pair]:
            pair_label_dict[pair]['0:perc'] = 0.
        if '1:perc' not in pair_label_dict[pair]:
            pair_label_dict[pair]['1:perc'] = 0.

    # usable_words = {word for word in word_label_dict if word_label_dict[word]['total'] >= count_threshold and abs(
    #     word_label_dict[word][str(1) + ":perc"] - word_label_dict[word][str(0) + ":perc"]) >= label_perc_diff_threshold}
    # 
    # usable_tags = {tag for tag in tag_label_dict if tag_label_dict[tag]['total'] >= count_threshold and abs(
    #     tag_label_dict[tag][str(1) + ":perc"] - tag_label_dict[tag][str(0) + ":perc"]) >= label_perc_diff_threshold}
    # 
    # usable_pairs = {pair for pair in pair_label_dict if pair_label_dict[pair]['total'] >= count_threshold and abs(
    #     pair_label_dict[pair][str(1) + ":perc"] - pair_label_dict[pair][str(0) + ":perc"]) >= label_perc_diff_threshold}
    # 
    # print(usable_words)
    # 
    # #TODO SUM LABEL 1 and LABEL 0-discriminating features separately [one sum for label 1 contributers and one sum for label 0 contributers]
    data_records = []
    for tagged_sentence, tagged_pairs in tqdm(zip(tagged_sentences, tagged_sentence_pairs), total=df.shape[0]):
        data_record = {'sum_1': 0, 'sum_0': 0, 'sum_none': 0}
        words, tags = zip(*tagged_sentence)
        pairs = [str(pair) for pair in tagged_pairs]
        usable_words_in_sentence = []
        for word in words:
            if word in word_label_dict:
                if word_label_dict[word]['0:perc'] > word_label_dict[word]['1:perc']:
                    data_record['sum_0'] += 1
                elif word_label_dict[word]['1:perc'] > word_label_dict[word]['0:perc']:
                    data_record['sum_1'] += 1
                else:
                    data_record['sum_none'] += 1
        for tag in tags:
            if tag in tag_label_dict:
                if tag_label_dict[tag]['0:perc'] > tag_label_dict[tag]['1:perc']:
                    data_record['sum_0'] += 1
                elif tag_label_dict[tag]['1:perc'] > tag_label_dict[tag]['0:perc']:
                    data_record['sum_1'] += 1
                else:
                    data_record['sum_none'] += 1

        for pair in pairs:
            if pair in pair_label_dict:
                if pair_label_dict[pair]['0:perc'] > pair_label_dict[pair]['1:perc']:
                    data_record['sum_0'] += 1
                elif pair_label_dict[pair]['1:perc'] > pair_label_dict[pair]['0:perc']:
                    data_record['sum_1'] += 1
                else:
                    data_record['sum_none'] += 1

        #     pairs = [str(pair) for pair in tagged_pairs]
        #     usable_words_in_sentence = []
        #     for word in words:
        #         if word in usable_words:
        #             if word not in data_record:
        #                 data_record[word] = 1
        #                 usable_words_in_sentence.append(word)
        #             else:
        #                 data_record[word] += 1
        #
        #     for tag in tags:
        #         if tag in usable_tags:
        #             if tag not in data_record:
        #                 data_record[tag] = 1
        #             else:
        #                 data_record[tag] += 1
        #
        #     for pair in pairs:
        #         if pair in usable_pairs:
        #             if pair not in data_record:
        #                 data_record[pair] = 1
        #             else:
        #                 data_record[pair] += 1

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

    return pd.DataFrame(data_records), tagged_sentences


if __name__ == "__main__":
    import pandas as pd
    import nltk
    from nltk import word_tokenize, pos_tag
    from tqdm import tqdm
    import argparse
    import reference_file_helpers as ref_helpers
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test_data_file', help='file containing test data')
    parser.add_argument(
        'aoa_data_file', help='file containing word Age of Acquisition Data')
    parser.add_argument(
        'concreteness_data_file', help='file containing word concreteness score data')
    parser.add_argument(
        'common_words_data_file', help='file containing most common word data')
    parser.add_argument(
        'word_eda_data_file', help='file containing word eda data')
    parser.add_argument(
        'tag_eda_data_file', help='file containing tag eda data')
    parser.add_argument(
        'pair_eda_data_file', help='file containing pair eda data')
    parser.add_argument(
        'vectorized_test_data_output_file', help='file to contain vectorized test data')
    parser.add_argument(
        'tagged_sentences_output_file', help='file to contain tagged sentences')
    args = parser.parse_args()
    aoa = ref_helpers.get_aoa_dataset(args.aoa_data_file)
    concreteness = ref_helpers.get_concreteness_dataset(args.concreteness_data_file)
    common_words = ref_helpers.get_common_words_dataset(args.common_words_data_file)
    transformed_data, tagged_sentences = get_transformed_data(args.test_data_file, aoa, concreteness,
                                                              common_words, args.word_eda_data_file,
                                                              args.tag_eda_data_file, args.pair_eda_data_file)

    transformed_data.to_pickle(args.vectorized_test_data_output_file)
    # labels.to_pickle(args.labels_output_file)
    pickle.dump(tagged_sentences, open(args.tagged_sentences_output_file, 'wb'))
