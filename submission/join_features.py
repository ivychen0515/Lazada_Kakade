import pandas as pd
import pickle
import numpy as np

f = open('counts.bin', 'rb')
counts = pickle.load(f)
score_df = pd.read_csv('../data/scores.csv', names = ['w2v_score', 'tf_score', 'sp_score'])
scores = score_df.values[1:]
feature = []
print(len(counts))
print(len(scores))
for i in range(len(counts)):
    x = counts[i]
    x = [float(i) for i in x ]
    y = scores[i]
    y = [float(i) for i in y ]
    x.extend(y)
    feature.append(x)

print(feature[0])
print(len(feature))

columns =  ['count_num',     'freq_num',
            'count_adj',     'freq_adj',
            'count_noun',    'freq_noun',
            'count_conj',    'freq_conj',
            'count_color',   'freq_color',
            'count_brand',   'freq_brand',
            'count_words',   'count_chars',
            'w2v_score', 'tf_score', 'sp_score']

feature_df = pd.DataFrame(feature, columns=columns)
print(feature_df.shape)
feature_df.to_pickle('final_features.bin')