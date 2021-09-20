# TODO: Visualize discriminative features using eda,eda_tags,eda_pairs (show as frequency vs label difference)
# TODO: Generate table for model results for different dataset configurations (show model hyperparameters used)
# TODO: Read through project requirements to make sure all concerns are answered
# TODO: Try BERT once vis are done?
# A - TFIDF with 5 Scores - best
# B - 5 Scores Only -
# C - Hand-selected features + 13 scores
# D - TFIDF + Hand-selected features + 13 scores (second_set.json files)
# E - TFIDF Summed + Hand-selected features Summed + 13 scores

# challenges during modeling:
# how to transform data to find most discriminative features
# how to handle large numbers of features and samples in memory
# how to
import pandas as pd

df = pd.read_pickle('../data/model/vectorized_train.pkl')
print(df.shape)
print(df.columns)
