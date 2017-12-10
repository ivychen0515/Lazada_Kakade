import pandas as pd
import tensorflow as tf
import numpy as np
from math import sqrt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

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

def neural_network(x, weights, biases):
	layer = tf.add(tf.matmul(x, weights['h']), biases['b'])
	out = tf.add(tf.matmul(layer, weights['out']), biases['out'])
	return out

def run_neural_network(num_input, num_classes, train_vec, train_labels,
	test_vec, test_labels, learning_rate=0.01):

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

	clarity_model = neural_network(X, weights, biases)
	prediction = tf.nn.softmax(clarity_model)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=clarity_model, labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss)

	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		original_test_labels = test_labels
		train_labels = [[1 - y, y] for y in train_labels]
		test_labels = [[1 - y, y] for y in test_labels]

		sess.run(train_op, feed_dict={X: train_vec, Y: train_labels})
		accu, nn_predict = sess.run([accuracy, tf.argmax(prediction, 1)], feed_dict={X: test_vec, Y: test_labels})
		f1 = f1_score(original_test_labels, nn_predict, average='macro')
		rmse = sqrt(mean_squared_error(original_test_labels, nn_predict))
		cm = np.array(confusion_matrix(original_test_labels, nn_predict), dtype='float')
		accu = accuracy_score(original_test_labels, nn_predict)
	return accu, f1, cm, rmse

def evaluate_accuracy(predict, truth):
	count = 0
	for i in range(len(predict)):
		if predict[i] == truth[i]:
			count += 1
	return float(count) / len(predict)

if __name__ == '__main__':
	category1, clarity, concise = getDataframe('../data/Clean_Data_Frame.csv')
	train_vec = read_vectors('../data/train_title_vectors')

	feat_vec = pd.read_pickle('../data/final_features.bin')
	drop_count = ['count_num', 'count_adj', 'count_noun', 'count_conj', 'count_color', 'count_brand', 'count_words', 'count_chars']
	drop_score = ['w2v_score', 'tf_score', 'sp_score']
	feat_vec = feat_vec.drop(drop_count + drop_score, axis=1)
	features = feat_vec.iloc[0]
	feat_vec = np.array(feat_vec.iloc[1:])
	
	categories = getCategories(category1)
	category_labels = assignCategoryIndex(category1, categories)
	train_labels = np.zeros(len(train_vec), dtype='int')
	for i in range(len(train_vec)):
		if category1[i] == 'Fashion' or category1[i] == 'Watches Sunglasses Jewellery':
			train_labels[i] = 1

	test_accuracy = 0
	test_f1 = 0
	test_rmse = 0
	test_cm = np.zeros((2,2))
	cv = cross_validation.KFold(len(train_vec), n_folds = 10)
	for traincv, testcv in cv:
		train_x = train_vec[traincv]
		train_y = train_labels[traincv]
		test_x = train_vec[testcv]
		test_y = train_labels[testcv]
	
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
		'''**********Random Forest**********'''
		'''
		rf_clf = RandomForestClassifier(max_features=50)
		rf_clf.fit(train_x, train_y)
		rf_predict = rf_clf.predict(test_x)
		test_f1 += f1_score(test_y, rf_predict, average='macro') / 5
		test_cm += np.array(confusion_matrix(test_y, rf_predict), dtype='float') / 5
		test_accuracy += accuracy_score(test_y, rf_predict) / 5
		'''

		'''**********NN Category**********'''
		'''
		accu, f1, cm, _ = run_neural_network(300, 2, train_vec[traincv], train_labels[traincv], train_vec[testcv], train_labels[testcv])
		test_f1 += f1 / 5
		test_cm += cm / 5
		test_accuracy += accu / 5
		'''

		'''**********NN Model A**********'''		
		'''
		accu, f1, cm, rmse = run_neural_network(len(feat_vec[0]), 2, feat_vec[A_idx], clarity[A_idx], feat_vec[test_A_idx], clarity[test_A_idx])
		test_f1 += f1 / 10
		test_cm += cm / 10
		test_rmse += rmse / 10
		test_accuracy += accu / 10
		'''
		'''**********Model B**********'''
		'''
		B_train_x = feat_vec[B_idx]
		B_train_y = clarity[B_idx]
		B_test_x = feat_vec[test_B_idx]
		B_test_y = clarity[test_B_idx]

		B_x_resampled = B_train_x.tolist()
		B_y_resampled = B_train_y.tolist()
		B_x_outlier = []
		B_y_outlier = []
		for i in range(len(B_train_x)):
			if B_train_y[i] == 0:
				B_x_outlier.append(B_train_x[i])
				B_y_outlier.append(B_train_y[i])

		for i in range(len(B_x_outlier)):
			for j in range(9):
				B_x_resampled.append(B_x_outlier[i])
				B_y_resampled.append(B_y_outlier[i])

		#outlier_detector = IsolationForest(n_estimators=200, contamination=0.02).fit(B_train_x)
		#outlier_detector = OneClassSVM().fit(B_train_x)
		#outlier_detector = LocalOutlierFactor().fit(B_train_x)
		outlier_detector = SVC().fit(B_x_resampled, B_y_resampled)
		outlier_predict = outlier_detector.predict(B_test_x)
		for i in range(len(outlier_predict)):
			if outlier_predict[i] == -1:
				outlier_predict[i] = 0
		
		test_rmse += sqrt(mean_squared_error(B_test_y, outlier_predict)) / 5
		test_cm += np.array(confusion_matrix(B_test_y, outlier_predict), dtype='float') / 5
		test_accuracy += accuracy_score(B_test_y, outlier_predict) / 5
		'''
	print("Testing Accuracy:", test_accuracy)
	print("Testing F1 Score:", test_f1)
	print("Testing RSME:", test_rmse)
	print("Testing Confusion Matrix:", test_cm)
	
	
		