import os

import pandas as pd
import numpy as np

# dataset gets
from tqdm import tqdm


def get_aoa_dataset(aoa_path):
    # get word/aoa pairs
    df = pd.read_csv(aoa_path, encoding='ISO-8859-1')[['Word', 'AoA_Kup_lem', 'Freq_pm', 'Perc_known_lem']].rename(
        columns={'AoA_Kup_lem': 'AoA'})
    df['Word'] = df['Word'].str.lower()
    return df


def get_common_words_dataset(common_words_path):
    # get list (python set) of common words
    return set(map(lambda w: w.lower().strip(), open(common_words_path, 'r').readlines()))


def get_concreteness_dataset(concreteness_path):
    # get word/mean concreteness rating pairs
    df = pd.read_csv(concreteness_path, sep='\t')[['Word', 'Conc.M', 'Percent_known']].rename(
        columns={'Conc.M': 'Concreteness'})
    df['Word'] = df['Word'].str.lower()
    return df


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
    data = [aoa.iloc[index[0][0]][['AoA', 'Freq_pm', 'Perc_known_lem']] for index in indices if index.size > 0]
    ages = [d['AoA'] for d in data]
    freq_pm = [d['Freq_pm'] for d in data]
    perc_known_lem = [d['Perc_known_lem'] for d in data]

    # print(indices,ages)
    # ages = [aoa[aoa['Word'] == word]['AoA'] for word in list_of_words]
    # ages = [age.iloc[0] for age in ages if not age.empty]
    return {
        'mean_age': np.nanmean(ages) if data else -1,
        'median_age': np.nanmedian(ages) if data else -1,
        'max_age': np.nanmax(ages) if data else -1,
        'mean_perc_known_lem': np.nanmean(perc_known_lem) if data else -1,
        'median_perc_known_lem': np.nanmedian(perc_known_lem) if data else -1,
        'mean_freq_pm': np.nanmean(freq_pm) if data else -1,
        'median_freq_pm': np.nanmedian(freq_pm) if data else -1,
    }


def get_count_of_uncommon_words(list_of_words, common_words):
    # return a sum of the counts of words in list of words that don't occur in common words (dale_chall.txt)
    word_count = 0
    for word in list_of_words:
        if word not in common_words:
            word_count += 1

    return word_count


def get_percentage_of_uncommon_words(list_of_words, common_words):
    # return a sum of the counts of words in list of words that don't occur in common words (dale_chall.txt)
    word_count = 0
    for word in list_of_words:
        if word not in common_words:
            word_count += 1

    return word_count / len(list_of_words)


def get_count_of_unique_common_words(list_of_words, common_words):
    # return a sum of binary 0/1 value for how many words a list_of_words contains
    # that are also in common_words (dale_chall.txt)
    pass


def get_dale_chall_stats(list_of_words, common_words):
    word_count = 0
    for word in list_of_words:
        if word not in common_words:
            word_count += 1
    l = len(list_of_words)
    return {
        'count_uncommon': word_count if word_count > 0 else -1,
        'perc_uncommon': word_count / l if l > 0 else -1
    }


def get_count_of_unknown_words(list_of_words, concreteness):
    # scores = [concreteness[concreteness['Word'] == word]['Percent_known'] for word in list_of_words]
    c = concreteness['Word'].to_numpy()
    indices = [np.argwhere(c == word) for word in list_of_words]
    # scores = [score.iloc[0] for score in scores if not score.empty]
    scores = [concreteness.iloc[index[0][0]]['Percent_known'] for index in indices if index.size > 0]
    return sum([1 for score in scores if score < 0.95])


def get_average_unknown_words_percentage(list_of_words, concreteness):
    c = concreteness['Word'].to_numpy()
    indices = [np.argwhere(c == word) for word in list_of_words]
    # scores = [score.iloc[0] for score in scores if not score.empty]
    scores = [concreteness.iloc[index[0][0]]['Percent_known'] for index in indices if index.size > 0]
    return np.nanmean(scores) if scores else -1


def get_lowest_unknown_words_percentage(list_of_words, concreteness):
    c = concreteness['Word'].to_numpy()
    indices = [np.argwhere(c == word) for word in list_of_words]
    # scores = [score.iloc[0] for score in scores if not score.empty]
    scores = [concreteness.iloc[index[0][0]]['Percent_known'] for index in indices if index.size > 0]
    return np.nanmin(scores) if scores else -1


def get_average_concreteness_rating(list_of_words, concreteness):
    # scores = [concreteness[concreteness['Word'] == word]['Concreteness'] for word in list_of_words]
    # scores = [score.iloc[0] for score in scores if not score.empty]
    c = concreteness['Word'].to_numpy()
    indices = [np.argwhere(c == word) for word in list_of_words]
    # scores = [score.iloc[0] for score in scores if not score.empty]
    scores = [concreteness.iloc[index[0][0]]['Concreteness'] for index in indices if index.size > 0]
    return np.nanmean(scores) if scores else -1


def get_lowest_concreteness_rating(list_of_words, concreteness):
    # scores = [concreteness[concreteness['Word'] == word]['Concreteness'] for word in list_of_words]
    # scores = [score.iloc[0] for score in scores if not score.empty]
    c = concreteness['Word'].to_numpy()
    indices = [np.argwhere(c == word) for word in list_of_words]
    # scores = [score.iloc[0] for score in scores if not score.empty]
    scores = [concreteness.iloc[index[0][0]]['Concreteness'] for index in indices if index.size > 0]
    return np.nanmin(scores) if scores else -1


def get_concreteness_stats(list_of_words, concreteness):
    c = concreteness['Word'].to_numpy()
    indices = [np.argwhere(c == word) for word in list_of_words]
    data = [concreteness.iloc[index[0][0]][['Concreteness', 'Percent_known']] for index in indices if index.size > 0]
    perc_known = [d['Percent_known'] for d in data]
    concreteness_ = [d['Concreteness'] for d in data]
    return {
        'mean_concreteness': np.nanmean(concreteness_) if data else -1,
        'min_concreteness': np.nanmin(concreteness_) if data else -1,
        'mean_perc_known': np.nanmean(perc_known) if data else -1,
        'min_perc_known': np.nanmin(perc_known) if data else -1,
    }


def get_words_in_sentence(list_of_words):
    return len(list_of_words)


vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}


def get_average_syllables_per_word(list_of_words):
    s = sum(1 if c in vowels else 0 for word in list_of_words for c in word)
    return s / len(list_of_words)


def get_maximum_syllables_in_one_word(list_of_words):
    s = (sum(1 if c in vowels else 0 for c in word) for word in list_of_words)
    return max(s)


def get_minimum_syllables_in_one_word(list_of_words):
    s = (sum(1 if c in vowels else 0 for c in word) for word in list_of_words)
    return min(s)


def get_word_stats(list_of_words):
    s = [sum(1. if c in vowels else 0 for c in word) for word in list_of_words]
    return {
        'mean_syllables': np.mean(s) if s else -1,
        'max_syllables': np.max(s) if s else -1,
        'min_syllables': np.min(s) if s else -1,
        'num_words': len(list_of_words)
    }


def get_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = None

    return (word for word in vocab)


functions_to_visualize = [

    get_percentage_of_uncommon_words, get_lowest_unknown_words_percentage,  # x
    get_count_of_uncommon_words, get_count_of_unknown_words,  # x
    get_average_concreteness_rating, get_lowest_concreteness_rating,  # x
    get_age_of_acquisition_stats,  # x
    get_average_syllables_per_word, get_maximum_syllables_in_one_word

]


def sentences_visualization(sentences, labels, aoa, concreteness, common_words, subset=10000, vis_df=None):
    if vis_df is None:
        random_indices = np.random.choice(range(len(sentences)), subset, replace=False)
        sentences = np.array(sentences)[random_indices]
        non_empty_indices = np.array([True if len(i) > 0 else False for i in sentences])
        sentences = sentences[non_empty_indices]
        labels = np.array(labels)[random_indices][non_empty_indices]
        print(sentences.shape, labels.shape)

        print("Getting 4 AOA scores...")
        # AOA
        aoa_stats = [get_age_of_acquisition_stats(sentence, aoa) for sentence in tqdm(sentences)]
        mean_aoa = [aoa_stat_dict['mean'] for aoa_stat_dict in tqdm(aoa_stats)]
        max_aoa = [aoa_stat_dict['max'] for aoa_stat_dict in tqdm(aoa_stats)]
        median_aoa = [aoa_stat_dict['median'] for aoa_stat_dict in tqdm(aoa_stats)]

        print("Getting 2 Dale Chall scores...")
        # Dale Chall
        perc_uncommon = [get_percentage_of_uncommon_words(sentence, common_words) for sentence in tqdm(sentences)]
        count_uncommon = [get_count_of_uncommon_words(sentence, common_words) for sentence in tqdm(sentences)]

        print("Getting 4 Concreteness scores...")
        # Concreteness
        avg_concreteness = [get_average_concreteness_rating(sentence, concreteness) for sentence in tqdm(sentences)]
        min_concreteness = [get_lowest_concreteness_rating(sentence, concreteness) for sentence in tqdm(sentences)]
        lowest_perc_unknown = [get_lowest_unknown_words_percentage(sentence, concreteness) for sentence in
                               tqdm(sentences)]
        count_unknown = [get_count_of_unknown_words(sentence, concreteness) for sentence in tqdm(sentences)]

        print("Getting 3 Syllable scores...")
        # Syllables
        average_syllables = [get_average_syllables_per_word(sentence) for sentence in tqdm(sentences)]
        max_syllables = [get_maximum_syllables_in_one_word(sentence) for sentence in tqdm(sentences)]
        min_syllables = [get_minimum_syllables_in_one_word(sentence) for sentence in tqdm(sentences)]

        print("Getting 1 Length score...")
        lengths = [get_words_in_sentence(sentence) for sentence in tqdm(sentences)]

        vis_df = pd.DataFrame({
            'mean_aoa': mean_aoa,
            'max_aoa': max_aoa,
            'median_aoa': median_aoa,
            'perc_uncommon': perc_uncommon,
            'count_uncommon': count_uncommon,
            'avg_concreteness': avg_concreteness,
            'min_concreteness': min_concreteness,
            'lowest_perc_unknown': lowest_perc_unknown,
            'count_unknown': count_unknown,
            'average_syllables': average_syllables,
            'max_syllables': max_syllables,
            'min_syllables': min_syllables,
            'lengths': lengths,
            'labels': labels

        })

        vis_df.to_pickle('vis_df.pkl')

    vis_df.corr().to_excel('vis_df_corr.xlsx')

    pd.plotting.scatter_matrix(vis_df)
    plt.show()


def establish_candidate_vocab(vocab, aoa, concreteness, common_words):
    # visualize: do common words correlate with sentences that don't need to be simplified? (percentage_of_uncommon_words vs label/target)
    # same idea but for concreteness (average concreteness, min concreteness)
    # same idea but for aoa (aoa stats mean,median, max)

    # take a small sample
    # establish basic relationships on variables before guessing what would work
    # then establish rules to only include certain words from vocab based on these relationships

    # try complex/compound sentences - number of transition words/conjunctions

    # look at which words are the most discriminative (hashvectorizer model vs labels)
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sentences = pd.read_pickle('../data/model/tokenized.pkl')
    labels = pd.read_pickle('../data/model/labels_lemma.pkl')
    print(list(zip(sentences[:5], labels[:5])))
    vis_df = None
    if os.path.exists('vis_df.pkl'):
        vis_df = pd.read_pickle('vis_df.pkl')
    vis = not True
    learn = True
    if vis:
        sentences_visualization(sentences, labels,
                                get_aoa_dataset('../data/reference/AoA_51715_words.csv'),
                                get_concreteness_dataset(
                                    '../data/reference/Concreteness_ratings_Brysbaert_et_al_BRM.txt'),
                                get_common_words_dataset('../data/reference/dale_chall.txt'), vis_df=vis_df,
                                subset=5000)

    if learn:
        print()
        from sklearn.model_selection import train_test_split

        print("No PCA")
        no_pca_scores = []
        from sklearn.ensemble import AdaBoostClassifier as clf_

        X = vis_df.loc[:, vis_df.columns != 'labels'].to_numpy()
        y = vis_df.loc[:, vis_df.columns == 'labels'].to_numpy().ravel()

        X, Xt, y, yt = train_test_split(X, y, test_size=0.2)
        clf = clf_()
        clf.fit(X, y)
        score = clf.score(Xt, yt)
        print(score)
        no_pca_scores.append(score)

        from sklearn.svm import LinearSVC as clf_

        clf2 = clf_()
        clf2.fit(X, y)
        score = clf2.score(Xt, yt)
        print(score)
        no_pca_scores.append(score)

        from sklearn.ensemble import GradientBoostingClassifier as clf_

        clf3 = clf_()
        clf3.fit(X, y)
        score = clf3.score(Xt, yt)
        print(score)
        no_pca_scores.append(score)

        from sklearn.ensemble import RandomForestClassifier as clf_

        clf4 = clf_()
        X = X[:, [i for i in range(vis_df.shape[1]) if i not in [6, 7, 8, 10, 11, 13]]]  # 6,7,8,10,11
        Xt = Xt[:, [i for i in range(vis_df.shape[1]) if i not in [6, 7, 8, 10, 11, 13]]]
        clf4.fit(X, y)
        score = clf4.score(Xt, yt)
        print(score)
        no_pca_scores.append(score)
        print(clf4.feature_importances_)

        from sklearn.ensemble import ExtraTreesClassifier as clf_

        clf5 = clf_()
        clf5.fit(X, y)
        score = clf5.score(Xt, yt)
        print(score)
        no_pca_scores.append(score)

        from sklearn.linear_model import LogisticRegression as clf_

        clf6 = clf_()
        clf6.fit(X, y)
        score = clf6.score(Xt, yt)
        print(score)
        no_pca_scores.append(score)

        print("With PCA")
        pca_scores = []
        from sklearn.decomposition import PCA
        from sklearn.ensemble import AdaBoostClassifier as clf_

        X_before = vis_df.loc[:, vis_df.columns != 'labels'].to_numpy()
        pca = PCA(n_components=5)
        X = pca.fit_transform(X_before)
        y = vis_df.loc[:, vis_df.columns == 'labels'].to_numpy().ravel()
        X, Xt, y, yt = train_test_split(X, y, test_size=0.2)
        clf = clf_()
        clf.fit(X, y)
        score = clf.score(Xt, yt)
        print(score)
        pca_scores.append(score)

        from sklearn.svm import LinearSVC as clf_

        clf2 = clf_()
        clf2.fit(X, y)
        score = clf2.score(Xt, yt)
        print(score)
        pca_scores.append(score)

        from sklearn.ensemble import GradientBoostingClassifier as clf_

        clf3 = clf_()
        clf3.fit(X, y)
        score = clf3.score(Xt, yt)
        print(score)
        pca_scores.append(score)

        from sklearn.ensemble import RandomForestClassifier as clf_

        clf4 = clf_()
        clf4.fit(X, y)
        score = clf4.score(Xt, yt)
        print(score)
        pca_scores.append(score)
        print(clf4.feature_importances_)

        from sklearn.ensemble import ExtraTreesClassifier as clf_

        clf5 = clf_()
        clf5.fit(X, y)
        score = clf5.score(Xt, yt)
        print(score)
        pca_scores.append(score)

        from sklearn.linear_model import LogisticRegression as clf_

        clf6 = clf_()
        clf6.fit(X, y)
        score = clf6.score(Xt, yt)
        print(score)
        pca_scores.append(score)

        print(no_pca_scores, np.mean(no_pca_scores))
        print(pca_scores, np.mean(pca_scores))
