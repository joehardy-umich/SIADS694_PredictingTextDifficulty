import pandas as pd
import numpy as np


# dataset gets
def get_aoa_dataset(aoa_path):
    # get word/aoa pairs
    return pd.read_csv(aoa_path, encoding='ISO-8859-1')[['Word', 'AoA_Kup_lem']].rename(columns={'AoA_Kup_lem': 'AoA'})


def get_common_words_dataset(common_words_path):
    # get list (python set) of common words
    return set(map(str.strip, open(common_words_path, 'r').readlines()))


def get_concreteness_dataset(concreteness_path):
    # get word/mean concreteness rating pairs
    return pd.read_csv(concreteness_path, sep='\t')[['Word', 'Conc.M', 'Percent_known']].rename(
        columns={'Conc.M': 'Concreteness'})


# helpers (assess a sentence for its scores)
# def get_average_age_of_acquisition(list_of_words, aoa):
#     ages = [aoa.str.match(word)['AoA'] for word in list_of_words]
#     return np.nanmean(ages)
#
#
# def get_median_age_of_acquisition(list_of_words, aoa):
#     ages = [aoa.str.match(word)['AoA'] for word in list_of_words]
#     return np.nanmedian(ages)

def get_age_of_acquisition_stats(list_of_words, aoa):
    a = aoa['Word'].to_numpy()
    indices = [np.argwhere(a == word) for word in list_of_words]
    # scores = [score.iloc[0] for score in scores if not score.empty]
    ages = [aoa.iloc[index[0][0]]['AoA'] for index in indices if index.size > 0]
    # print(indices,ages)
    # ages = [aoa[aoa['Word'] == word]['AoA'] for word in list_of_words]
    # ages = [age.iloc[0] for age in ages if not age.empty]
    return {
        'mean': np.nanmean(ages) if ages else -1,
        'median': np.nanmedian(ages) if ages else -1
    }


def get_count_of_uncommon_words(list_of_words, common_words):
    # return a sum of the counts of words in list of words that occur in common words (dale_chall.txt)
    word_count = 0
    for word in list_of_words:
        if word not in common_words:
            word_count += 1

    return word_count


def get_count_of_unique_common_words(list_of_words, common_words):
    # return a sum of binary 0/1 value for how many words a list_of_words contains
    # that are also in common_words (dale_chall.txt)
    pass


def get_count_of_unknown_words(list_of_words, concreteness):
    # scores = [concreteness[concreteness['Word'] == word]['Percent_known'] for word in list_of_words]
    c = concreteness['Word'].to_numpy()
    indices = [np.argwhere(c == word) for word in list_of_words]
    # scores = [score.iloc[0] for score in scores if not score.empty]
    scores = [concreteness.iloc[index[0][0]]['Percent_known'] for index in indices if index.size > 0]
    return sum([1 for score in scores if score < 0.95])


def get_average_concreteness_rating(list_of_words, concreteness):
    # scores = [concreteness[concreteness['Word'] == word]['Concreteness'] for word in list_of_words]
    # scores = [score.iloc[0] for score in scores if not score.empty]
    c = concreteness['Word'].to_numpy()
    indices = [np.argwhere(c == word) for word in list_of_words]
    # scores = [score.iloc[0] for score in scores if not score.empty]
    scores = [concreteness.iloc[index[0][0]]['Concreteness'] for index in indices if index.size > 0]
    return np.nanmean(scores) if scores else -1
