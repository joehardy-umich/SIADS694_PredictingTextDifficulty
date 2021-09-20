def get_all_tfidf(D):
    all_scores = []
    for d in D:
        scores = []
        for t in d:
            scores.append(tfidf(t, d, D))
        all_scores.append(scores)

    return all_scores


def tfidf(t, d, D):
    return tf(t, d) * idf(t, D)


def tf(t, d):
    f = sum(1 for d_t in d if d_t == t)
    return f / len(d)


def idf(t, D):
    t_in_d_for_d_in_D = sum(1 for d in D if t in d)
    return len(D) / (t_in_d_for_d_in_D + 1)


docs = [[8, 6, 7, 9, 5, 4, 7, 3, 2], [8, 9, 2, 2, 3, 4], [1, 3, 5, 10], [2, 4, 7, 8, 9], [1, 2, 3, 4, 6]]

tfidfs = get_all_tfidf(docs)
print(list(sum(tfidf_) for tfidf_ in tfidfs))
