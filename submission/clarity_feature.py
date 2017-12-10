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
    w2v = gensim.models.Word2Vec(size=200, min_count=1, window=5)
    w2v.build_vocab(dictionary)
    w2v.train(dictionary, total_examples=w2v.corpus_count, epochs=w2v.iter)
    w2v.save('../data/new_w2v.mdl')

    # w2v = gensim.models.Word2Vec.load('../data/new_w2v.mdl')
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
    
def get_type():
    type_set = set()
    data = read_data('../data/Title_Word_entity.bin')
    for title in data:
        for item in title:
            type_set.add(item[0])
    print(type_set)
    return type_set
    

def count_feature():
    data = read_data('../data/Title_Word_entity.bin')
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
            if item[1] in brand_dict:
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

def count_title_des_dup(title, des):
    count = []
    for i in range(len(title)):
        line = 0
        inter = set(title[i]).intersection(set(des[i]))
        count.append(len(inter))
    return count

def join_feature(counts, scores):
    feature = []
    print(len(counts))
    scores = np.array(scores).T.tolist()
    print(len(scores))
    for i in range(len(counts)):
        x = counts[i]
        x = [float(i) for i in x ]
        y = scores[i]
        y = [float(i) for i in y ]
        x.extend(y)
        feature.append(x)
    print(len(feature[0]))
    print(len(feature))

    columns =  ['count_num',     'freq_num',
                'count_adj',     'freq_adj',
                'count_noun',    'freq_noun',
                'count_conj',    'freq_conj',
                'count_color',   'freq_color',
                'count_brand',   'freq_brand',
                'count_words',   'count_chars',
                'max_words_topic', 'max_words_topic_freq', 'topic_num',
                'title_cat_dup', 'title_des_dup', 'w2v_score', 'tf_score', 'sp_score']
    
    feature_df = pd.DataFrame(feature, columns=columns)
    print(feature_df.shape)
    feature_df.to_pickle('../data/final_features.bin')

if __name__ == '__main__':
    # title = read_data('../data/raw_title.bin')
    # des = read_data('../data/raw_des.bin')

    infile = pd.read_json('../data/Dataframe_Of_Everything.json', lines=True)
    title = infile['new_title_stem'].tolist() 
    des = infile['new_des_stem'].tolist()
    title_cat_dup = infile['result'].tolist()
    infile2 = pd.read_json('../data/LDA.json', lines=True)
    lda = infile2['new'].tolist()
    max_words_topic = []
    topic_num = []
    for t in lda:
        max_words_topic.append(int(t[0]))
        topic_num.append(int(t[1]))

    print('Word2Vec calculation')
    # w2v_score = train_word2vec(title, des)
    # print(w2v_score)
    # with open('../data/w2v_score.bin', 'wb') as f:
    #     pickle.dump(w2v_score, f)
    with open('../data/w2v_score.bin', 'rb') as f:
        w2v_score=pickle.load(f)

    print('Spacy similarity calculation')
    # sp_score = spacy_similarity(title, des)
    # print(sp_score)
    # with open('../data/sp_score.bin', 'wb') as f:
    #     pickle.dump(sp_score, f)
    with open('../data/sp_score.bin', 'rb') as f:
        sp_score=pickle.load(f)

    print('TF*IDF similarity calculation')
    # tf_score = tfidf_similarity(title, des)
    # print(tf_score)
    # with open('../data/tf_score.bin', 'wb') as f:
    #     pickle.dump(tf_score, f)
    with open('../data/tf_score.bin', 'rb') as f:
        tf_score=pickle.load(f)

    title_des_dup = count_title_des_dup(title, des)
    counts = count_feature()
    words_length = np.array(counts)[:,12].tolist()
    total_scores = [max_words_topic, list(map(lambda x, y: x/y, max_words_topic, words_length)) , topic_num, title_cat_dup, title_des_dup, w2v_score, tf_score, sp_score]
    join_feature(counts, total_scores)
    
    # get_type()
