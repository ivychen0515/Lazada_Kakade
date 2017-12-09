import re
import pandas as pd
import spacy
import gensim
import numpy as np
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle

def special_char(title):
    s_char = []
    for l in title:
        res = re.findall('', l, re.U)
        s_char.append(len(res))
    return s_char

def read_data(path):
    raw_data = pd.read_pickle(path)
    raw_data = raw_data.values.tolist()
    list_data = []
    for i in raw_data:
        i = [x for x in i if x != None]
        list_data.append(i)
    return list_data

def train_word2vec(title, des):
    dictionary = []
    for i in range(len(title)):
        a = title[i]
        a.extend(des[i])
        dictionary.append(a)
    # w2v = gensim.models.Word2Vec(size=200, min_count=1, window=5)
    # w2v.build_vocab(dictionary)
    # w2v.train(dictionary, total_examples=w2v.corpus_count, epochs=w2v.iter)
    # w2v.save('new_w2v.mdl')

    w2v = gensim.models.Word2Vec.load('new_w2v.mdl')
    score = []
    for i in range(len(title)):
        line = []
        for x in title[i]:
            for y in des[i]:
                try:
                    l = w2v.similarity(x, y)
                except:
                    print(x,' & ',y,'does not exist')
                else:
                    if np.isnan(l):
                        line.append(0)
                    else:
                        line.append(l)
        try:
            s = np.mean(line)
        except RuntimeWarning:
            print(i)
            print(x)
            print(y)
        else:
            score.append(s)

    return score

def spacy_similarity(title, des):
    nlp = spacy.load('en')
    score = []
    for i in range(len(title)):
        t = ' '.join(title[i])
        d = ' '.join(des[i])
        t1 = nlp(t)
        d1 = nlp(d)
        score.append(t1.similarity(d1))
    return score

def tfidf_similarity(title, des):
    dictionay = []
    for i in range(len(title)):
        t = ' '.join(title[i])
        d = ' '.join(des[i])
        dictionay.append(t)
        dictionay.append(d)
    tfidf = TfidfVectorizer().fit_transform(dictionay)
    score = []
    for i in range(len(title)):
        j = i * 2
        score.append(linear_kernel(tfidf[j], tfidf[j+1]).flatten()[0])
    return score

def get_brand():
    f1 = open('../data/brand.txt', encoding='latin1')
    f2 = open('../data/brands_from_lazada_portal.txt', encoding='latin1')
    c1 = f1.readlines()
    c1 = [x.strip() for x in c1]
    c2 = f1.readlines()
    c2 = [x.strip() for x in c2]
    s = set(c1)
    s1 = set(c2)
    s.update(s1)
    return s

def get_color():
    f1 = open('../data/title_colors.txt', encoding='latin1')
    f2 = open('../data/desc_colors.txt', encoding='latin1')
    c1 = f1.readlines()
    c1 = [x.strip() for x in c1]
    c2 = f1.readlines()
    c2 = [x.strip() for x in c2]
    s = set(c1)
    s1 = set(c2)
    s.update(s1)
    return s

def count_feature():
    data = read_data('../data/raw.bin')
    color_dict = get_color()
    brand_dict = get_brand()
    total_counts = []
    for title in data:
        count_num = 0
        count_adj = 0
        count_conj = 0
        count_color = 0
        count_brand = 0
        count_noun = 0
        count_words = len(title)
        count_chars = 0
        for item in title:
            count_chars += len(item[1])
            if item[0] == 'NUM':
                count_num += 1
            if item[0] == 'ADJ':
                count_adj += 1
            if item[0] == 'NOUN' or item[0] == 'PROPN':
                count_noun += 1
            if item[0] == 'SYM' or item[0] == 'CCONJ':
                count_conj += 1
            if item[1] in color_dict:
                count_color += 1
            if item[1] in color_dict:
                count_brand += 1
        line = [count_num, count_num/count_words,
            count_adj, count_adj/count_words,
            count_noun, count_noun/count_words,
            count_conj, count_conj/count_words,
            count_color, count_color/count_words,
            count_brand, count_brand/count_words,
            count_words, count_chars]
        total_counts.append(line)
    return total_counts

def join_feature():
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

if __name__ == '__main__':
    # title = read_data('../data/raw_title.bin')
    # des = read_data('../data/raw_des.bin')
    
    # print('Word2Vec calculation')
    # w2v_score = train_word2vec(title, des)
    # print(w2v_score)
    # w2v_score = pd.read_csv('../data/w2v_score.csv', header=None)
    # w2v_score = w2v_score.values.flatten().tolist()

    # print('Spacy similarity calculation')
    # sp_score = spacy_similarity(title, des)
    # print(sp_score)
    # with open('sp_score.bin', 'wb') as f:
    #     pickle.dump(sp_score, f)

    # print('TF*IDF similarity calculation')
    # tf_score = tfidf_similarity(title, des)
    # print(tf_score)
    # tf_score = pd.read_csv('../data/tf_score.csv', header=None)
    # tf_score = tf_score.values.flatten().tolist()

    total_scores = [w2v_score, tf_score, sp_score]

    counts = count_feature()
    with open('../data/counts.bin', 'wb') as f:
        pickle.dump(counts, f)
