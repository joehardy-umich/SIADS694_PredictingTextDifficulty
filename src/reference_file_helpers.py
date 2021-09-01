# transforms
def get_aoa_dataset(aoa_path):
    # get word/aoa pairs
    pass


def get_common_words_dataset(common_words_path):
    # get list (python set) of common words
    pass


def get_concreteness_dataset(concreteness_path):
    # get word/mean concreteness rating pairs
    pass


# helpers (assess a sentence for its scores)
def get_average_age_of_acquisition(list_of_words, aoa):
    pass


def get_median_age_of_acquisition(list_of_words, aoa):
    pass


def get_count_of_common_words(list_of_words, common_words):
    # return a sum of the counts of words in list of words that occur in common words (dale_chall.txt)
    pass


def get_count_of_unique_common_words(list_of_words, common_words):
    # return a sum of binary 0/1 value for how many words a list_of_words contains
    # that are also in common_words (dale_chall.txt)
    pass


def average_concreteness_rating(list_of_words, concreteness):
    pass
