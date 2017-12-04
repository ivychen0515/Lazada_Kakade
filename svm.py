import pandas as pd
<<<<<<< HEAD
import gensim
import numpy as np
from sklearn import svm
=======
#import gensim
import numpy as np
import string
import ast
from sklearn import svm
from sklearn import cross_validation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
>>>>>>> 0d8dd22dcf37c5df534c24dfcfbe2f1937fda4f2

def getDataframe(filePath):
	dataframe = pd.read_csv(filePath)
	concise = dataframe['concise']
	clarity = dataframe['clarity']
	x = dataframe.drop(['concise', 'clarity', 'index'], axis=1)
	return x, clarity, concise

def title_vec(title, w2v_model):
<<<<<<< HEAD
	title = title.split(' ')
	vector = np.zeros(300)
	for word in title:
=======
	stop_words = set(stopwords.words('english'))
	translator = str.maketrans(dict.fromkeys(string.punctuation))
	title = title.translate(translator)
	word_tokens = word_tokenize(title)
	words = [w.lower() for w in word_tokens if not w in stop_words]

	vector = np.zeros(300)
	for word in words:
>>>>>>> 0d8dd22dcf37c5df534c24dfcfbe2f1937fda4f2
		try:
			vector += w2v_model[word]
		except:
			pass
	return vector

<<<<<<< HEAD
def evaluate_prediction(predict, test):
	count = 0
	for i in range(len(predict)):
		if predict[i] == test[i]:
			count += 1 
	print("Precision: " + str(float(count) / len(predict)))

if __name__ == '__main__':
	path = './Clean_Data_Frame.csv'
	x, clarity, concise = getDataframe(path)
	w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../Support_files/GoogleNews-vectors-negative300.bin', binary=True)
	title_vectors = [title_vec(title, w2v_model) for title in x['title']]
	out_file = open('title_vectors', 'w', encoding='utf-8')
	for vec in title_vectors:
		out_file.write(str(vec))
	out_file.write('\n')
	out_file.close()
	#clf = svm.SVC()
	#clf.fit(title_vectors, clarity)
	#evaluate_prediction(clf.predict(test_feat_vectors), test_labels)

=======
def evaluate_accuracy(predict, test):
	count = 0
	for i in range(len(predict)):
		if predict[i] == test[i]:
			count += 1
	return float(count) / len(predict)

def read_vectors(filename):
	in_file = open(filename, 'r')
	title_vectors = []
	for line in in_file:
		title_vectors.append(list(map(float, line.rstrip().split(' '))))
	in_file.close()
	return title_vectors

def write_vectors(filename, title_vectors):
	out_file = open(filename, 'w', encoding='utf-8')
	for vec in title_vectors:
		for x in vec:
			out_file.write(str(x))
			out_file.write(' ')
		out_file.write('\n')
	out_file.close()

if __name__ == '__main__':
	path = './Clean_Data_Frame.csv'
	train_x, train_clarity, train_concise = getDataframe(path)
    
    #w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    #desc_vectors = [title_vec(str(title), w2v_model) for title in train_x['description']]
    #title_vectors = [title_vec(title, w2v_model) for title in x['title']]
    #write_vectors('description_vector', desc_vectors)
    
	train_title_vectors = np.array(read_vectors('train_title_vectors'))
	train_clarity = np.array(train_clarity)
	cv = cross_validation.KFold(len(train_title_vectors), n_folds = 5)
	accuracy = 0
	for traincv, testcv in cv:
		train_x = train_title_vectors[traincv]
		train_y = train_clarity[traincv]
		test_x = train_title_vectors[testcv]
		test_y = train_clarity[testcv]
		clf = svm.SVC()
		clf.fit(train_x, train_y)
		accuracy += evaluate_accuracy(clf.predict(test_x), test_y) / 10
	print(accuracy)
>>>>>>> 0d8dd22dcf37c5df534c24dfcfbe2f1937fda4f2
