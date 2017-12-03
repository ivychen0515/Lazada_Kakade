import numpy as np
import pickle
import gensim
import pandas as pd
import re
import unicodedata
from sklearn.linear_model import LogisticRegression as lg
from sklearn import model_selection

def read_vectors(filename):
    in_file = open(filename, 'r')
    wordVectors = []
    for line in in_file:
        wordVectors.append(list(map(float, line.rstrip().split(' '))))
    in_file.close()
    return wordVectors

def read_file(filename):
    in_file = pd.read_csv(filename)
    row, _ = in_file.shape
    dictionary = []
    title = []
    descr = []
    clarity = in_file['clarity']
    for i in range(row):
        dfl = in_file.iloc[i]
        tl = str(dfl['title'])
        dl = str(dfl['description'])
        tl = sentence = re.split(',| |\(|\)|-|:|\&|\.|\+|;|/|"', unicodedata.normalize('NFKD', tl))
        dl = sentence = re.split(',| |\(|\)|-|:|\&|\.|\+|;|/|"', unicodedata.normalize('NFKD', dl))
        tl = [x for x in tl if x != None and x != '']
        dl = [x for x in dl if x != None and x != '']
        sentence = []
        sentence.extend(tl)
        sentence.extend(dl)
        title.append(tl)
        descr.append(dl)
        dictionary.append(sentence)
    return dictionary, title, descr, clarity

def generate_w2v_model(filename, file_dict):
    w2v = gensim.models.Word2Vec(size=200, min_count=1)
    w2v.build_vocab(file_dict)
    print('start training')
    w2v.train(file_dict, total_examples=w2v.corpus_count, epochs=w2v.iter)
    w2v.save(filename)
    return w2v

def distribution(test): 
    one = 0
    zero = 0
    for i in test:
        if i == 0:
            zero += 1
        if i == 1:
            one += 1
    print('one: ',one,'\tzero: ',zero)
    return one, zero

def cal_similarity(model, title, descr):
    similarity = []
    outlier = []
    nan = []
    for i in range(len(title)):
        line = []
        for x in title[i]:
            one_word = []
            for y in descr[i]:
                try:
                    l = model.similarity(x, y)
                except KeyError:
                    print(x,' & ',y,'does not exist')
                else:
                    one_word.append(l)
            one_word.sort(reverse=True)
            if (len(title[i])<5):
                line.extend(one_word)
            else:
                one_word = one_word[0:5]
                line.append(np.mean(one_word))
        while len(line) < 5:
            line.append(0)
        if len(line) < 5:
            outlier.append(i)
        line.sort(reverse=True)
        for x in line:
            if (np.isnan(x)):
                nan.append(i)
                break
        similarity.append(line[0:10])
    # print(list(map(lambda x: np.mean(x), similarity)))
    print('titles has less than 5 words:')
    for i in outlier:
        print(i,': ',title[i])
    print('\ntitles has NaN:')    
    for i in nan:
        print(i,': ',title[i])
    return similarity

if __name__ == "__main__":
    ## read title vector
    # path_titleVector = "../train_title_vectors.txt"
    # title_vectors = read_vectors(path_title_vector)

    ## read clean raw data
    path_dataFrame = "../Clean_Data_Frame.csv"
    dictionary, title, descr, clarity = read_file(path_dataFrame)

    ## generate word2vec model
    # path_model = 'w2v_model.txt'
    # w2v_model = generate_w2v_model(path_model, dictionary)

    ## load word2vec model and calculate similarity
    w2v_model = gensim.models.Word2Vec.load('w2v_model.txt')
    similarity = cal_similarity(w2v_model, title, descr)
    ts = open('similarity.txt','wb')
    pickle.dump(similarity, ts)

    ## data preparation for training
    train_title = np.array(similarity)
    train_clarity = np.array(clarity)
    np.savetxt('training_similarity.npy', train_title)
    np.savetxt('training_clarity.npy', train_clarity)

    ## cross validation preparation
    # train_title = np.loadtxt('training_similarity.npy')
    # train_clarity = np.loadtxt('training_clarity.npy')
    # cv = model_selection.KFold(10)
    # for i in range(len(train_title)):
    #     for j in range(5):
    #         if np.isnan(train_title[i][j]):
    #             train_title[i][j] = 0
    # accuracy = 0

    ## start training logistic regression model
    # for train_n, test_n in cv.split(train_title):
    #     train_x = train_title[train_n]
    #     train_y = train_clarity[train_n]
    #     test_x = train_title[test_n]
    #     test_y = train_clarity[test_n]
    #     l_model = lg()
    #     l_model.fit(train_x, train_y)
    #     predicted_y = l_model.predict(test_x)
    #     mse = list(map(lambda x,y: (x-y)*(x-y), test_y, predicted_y))
    #     accuracy += np.sqrt(np.mean(mse))
    # print(accuracy/cv.get_n_splits())