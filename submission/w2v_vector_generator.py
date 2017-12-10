import pandas as pd
import gensim
import numpy as np
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def getDataframe(filePath):
	dataframe = pd.read_csv(filePath)
	concise = dataframe['concise']
	clarity = dataframe['clarity']
	x = dataframe.drop(['concise', 'clarity', 'index'], axis=1)
	return x, clarity, concise

def title_vec(title, w2v_model):
	vector = np.zeros(300)
	stop_words = set(stopwords.words('english'))
	translator = str.maketrans(dict.fromkeys(string.punctuation))
	title = title.translate(translator)
	word_tokens = word_tokenize(title)
	words = [w.lower() for w in word_tokens if not w in stop_words]

	vector = np.zeros(300)

	for word in words:
		try:
			vector += w2v_model[word]
		except:
			pass

	return vector

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
	w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
	desc_vectors = [title_vec(str(title), w2v_model) for title in train_x['description']]
	title_vectors = [title_vec(title, w2v_model) for title in train_x['title']]
	write_vectors('description_vector', desc_vectors)
	write_vectors('title_vector', title_vectors)
