import pandas as pd
import gensim
import numpy as np
from sklearn import svm

def getDataframe(filePath):
	dataframe = pd.read_csv(filePath)
	concise = dataframe['concise']
	clarity = dataframe['clarity']
	x = dataframe.drop(['concise', 'clarity', 'index'], axis=1)
	return x, clarity, concise

def title_vec(title, w2v_model):
	title = title.split(' ')
	vector = np.zeros(300)
	for word in title:
		try:
			vector += w2v_model[word]
		except:
			pass
	return vector

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

