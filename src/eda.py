import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from tqdm import tqdm
nltk.download('averaged_perceptron_tagger')
tqdm.pandas()
df = pd.read_csv('../data/model/WikiLarge_Train.csv')
print(df.columns)
labels = df['label']
tokenized_text = df['original_text'].progress_apply(word_tokenize)
word_label_dict = {}
tag_label_dict = {}
pair_label_dict={}
for sentence, label in tqdm(zip(tokenized_text, labels)):
    tagged_sentence = pos_tag(sentence)
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
    if len(tagged_sentence)>1:
        tag_pairs = [[tag[1] for tag in tagged_sentence[i:i+2]] for i in range(0,len(tagged_sentence)-2)]
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
for word in tqdm(word_label_dict):
    sum_labels = sum(word_label_dict[word].values())
    for label in list(word_label_dict[word].keys()):
        word_label_dict[word][str(label) + ":perc"] = word_label_dict[word][label] / sum_labels

for tag in tqdm(tag_label_dict):
    sum_labels = sum(tag_label_dict[tag].values())
    for label in list(tag_label_dict[tag].keys()):
        tag_label_dict[tag][str(label) + ":perc"] = tag_label_dict[tag][label] / sum_labels
        
for pair in tqdm(pair_label_dict):
    sum_labels = sum(pair_label_dict[pair].values())
    for label in list(pair_label_dict[pair].keys()):
        pair_label_dict[pair][str(label) + ":perc"] = pair_label_dict[pair][label] / sum_labels

word_df = pd.DataFrame.from_records(word_label_dict)
word_df.T.to_csv('eda_words.csv')

tag_df = pd.DataFrame.from_records(tag_label_dict)
tag_df.T.to_csv('eda_tags.csv')

pair_df = pd.DataFrame.from_records(pair_label_dict)
pair_df.T.to_csv('eda_pairs.csv')
