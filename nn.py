import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

def applyZScore(dataframe):
    normalized_dataframe = dataframe
    normalized_dataframe = (normalized_dataframe 
        - normalized_dataframe.mean(axis=0)) / normalized_dataframe.std(axis=0)
    return normalized_dataframe

def read_vectors(filename):
    in_file = open(filename, 'r')
    title_vectors = []
    for line in in_file:
        title_vectors.append(list(map(float, line.rstrip().split(' '))))
    in_file.close()
    return np.array(title_vectors)

def getDataframe(filePath):
	dataframe = pd.read_csv(filePath)
	concise = np.array([label for label in dataframe['concise']])
	clarity = np.array([label for label in dataframe['clarity']])
	#x = dataframe.drop(['concise', 'clarity', 'index'], axis=1)

	return dataframe['category_1'], clarity, concise

def getCategories(data):
	cate = set()
	for c in data:
		cate.add(c)
	return list(cate)

def assignCategoryIndex(data, categories):
	category_labels = np.zeros((len(data), len(categories)))
	for i in range(len(data)):
		category_labels[i][categories.index(data[i])] = 1
	return category_labels

def neural_network(x):
	layer = tf.add(tf.matmul(x, weights['h']), biases['b'])
	out = tf.add(tf.matmul(layer, weights['out']), biases['out'])
	return out

def evaluate_accuracy(predict, truth):
	count = 0
	for i in range(len(predict)):
		if predict[i] == truth[i]:
			count += 1
	return float(count) / len(predict)

if __name__ == '__main__':
	category1, clarity, concise = getDataframe('./Clean_Data_Frame.csv')
	train_vec = read_vectors('./train_title_vectors')

	feat_vec = pd.read_pickle('./final_features.bin')
	drop_count = ['count_num', 'count_adj', 'count_noun', 'count_conj', 'count_color', 'count_brand', 'count_words', 'count_chars']
	drop_score = ['w2v_score', 'tf_score', 'sp_score']
	feat_vec = feat_vec.drop(drop_count + drop_score, axis=1)
	#feat_vec = feat_vec[drop_score]
	features = feat_vec.iloc[0]
	feat_vec = np.array(applyZScore(feat_vec.iloc[1:]))
	
	categories = getCategories(category1)
	category_labels = assignCategoryIndex(category1, categories)
	train_labels = np.zeros(len(train_vec))
	for i in range(len(train_vec)):
		if category1[i] == 'Fashion' or category1[i] == 'Watches Sunglasses Jewellery':
			train_labels[i] = 1

	test_accuracy = 0
	cv = cross_validation.KFold(len(train_vec), n_folds = 5)
	for traincv, testcv in cv:
		train_x = train_vec[traincv]
		train_y = train_labels[traincv]
		test_x = train_vec[testcv]
		test_y = train_labels[testcv]
		nn_vec = np.zeros((len(test_y), 3))
	
		A_idx = []
		B_idx = []
		for i in traincv:
			if category1[i] == 'Fashion' or category1[i] == 'Watches Sunglasses Jewellery':
				A_idx.append(i)
			else:
				B_idx.append(i)

		test_A_idx = []
		test_B_idx = []
		for i in testcv:
			if category1[i] == 'Fashion' or category1[i] == 'Watches Sunglasses Jewellery':
				test_A_idx.append(i)
			else:
				test_B_idx.append(i)

		# Model A Label : 1, Model B Label 0
		rf_clf = RandomForestClassifier(max_depth=2, random_state=0)
		rf_clf.fit(train_x, train_y)
		rf_predict = rf_clf.predict_proba(test_x)

		'''**********Model A**********'''

		learning_rate = 0.9

		num_input = len(feat_vec[0])
		num_classes = 2
		n_hidden = int((num_input + num_classes) / 2)

		X = tf.placeholder('float', [None, num_input])
		Y = tf.placeholder('float', [None, num_classes])

		weights = {
			'h': tf.Variable(tf.random_normal([num_input, n_hidden])),
			'out': tf.Variable(tf.random_normal([n_hidden, num_classes]))
		}

		biases = {
			'b': tf.Variable(tf.random_normal([n_hidden])),
			'out': tf.Variable(tf.random_normal([num_classes]))
		}

		clarity_model = neural_network(X)
		prediction = tf.nn.softmax(clarity_model)

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=clarity_model, labels=Y))
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(loss)

		correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
		false_positive = tf.reduce_sum(tf.multiply(tf.argmax(prediction, 1), 1 - tf.argmax(Y, 1)))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		

		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)
			
			A_train_x = feat_vec[A_idx]
			A_train_y = clarity[A_idx]
			A_train_y = [[1 - y, y] for y in A_train_y]
			A_test_x = feat_vec[test_A_idx]
			A_test_y = clarity[test_A_idx]
			A_test_y = [[1 - y, y] for y in A_test_y]
			#mat = tf.confusion_matrix(labels=A_test_y, predictions=prediction, num_classes=2)

			c, _ = sess.run([loss, train_op], feed_dict={X: A_train_x, Y: A_train_y})
			print("NN Loss: ", c)
			#matrix = sess.run(mat)
			#print(matrix)
			accu = sess.run(accuracy, feed_dict={X: A_test_x, Y: A_test_y})
			print("NN: ", accu)
			test_accuracy += float(accu) / 5

		'''**********Model B**********'''
		B_train_x = feat_vec[B_idx]
		B_train_y = clarity[B_idx]
		B_test_x = feat_vec[test_B_idx]
		B_test_y = clarity[test_B_idx]
		ocsvm = IsolationForest().fit(B_train_x, B_train_y)
		outlier_predict = ocsvm.predict(B_test_x)
		for i in range(len(outlier_predict)):
			if outlier_predict[i] == -1 :
				outlier_predict[i] = 0
		print("Outlier: ", evaluate_accuracy(outlier_predict, B_test_y))

		'''
		clarity_score = np.zeros((len(testcv), 2))
		for i in range(len(testcv)):
			clarity_score[i][0] = modelA_score[0] * rf_predict[i][1] + modelB_score[0] * rf_predict[i][0]
			clarity_score[i][1] = modelA_score[1] * rf_predict[i][1] + modelB_score[1] * rf_predict[i][0]
		'''

	print("Testing Accuracy:", test_accuracy)
	
	
		