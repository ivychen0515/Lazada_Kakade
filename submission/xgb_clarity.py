import pandas as pd
import numpy as np
from xgboost import XGBClassifier as xgbc
from xgboost import XGBRegressor as xgbr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lg
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture as gmm
from sklearn.mixture import BayesianGaussianMixture as bgmm
from collections import Counter 

infile = pd.read_csv('../data/Clean_Data_Frame.csv')
infile2 = pd.read_json('../data/Dataframe_Of_Everything.json', lines=True)
row, _ = infile.shape
cat = []
cat2 = []
for i in range(row):
    dfl = infile.iloc[i]
    if str(dfl['category_1']) == 'Fashion' or str(dfl['category_1']) == 'Watches Sunglasses Jewellery':
    # if str(dfl['category_1']) != 'Fashion' and str(dfl['category_1']) != 'Watches Sunglasses Jewellery':
        cat.append(i)
    else:
        cat2.append(i)
clarity = infile['clarity']
feature = pd.read_pickle('final_features.bin').drop(['w2v_score'], axis=1)

def test_model(model, cat):
    X = feature.values[cat]
    Y = clarity[cat]
    seed = 56 
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    for i in np.arange(0.5, 1, 0.1):
        predictions = []
        one = 0
        zero = 0
        for item in y_pred:
            if item > i:
                predictions.append(1)
                one += 1
            else:
                predictions.append(0)
                zero += 1
    
        accuracy = accuracy_score(y_test, predictions)
        print("%.4f: Accuracy: %.2f%%" % (i, accuracy * 100.0))
        print('zero percentage %.4f' % (zero / (zero + one)))
        mse = list(map(lambda x,y: (x-y)*(x-y), y_test, y_pred))
        print("RMSE: %.2f" % (np.sqrt(np.mean(mse))))
        print(confusion_matrix(y_test, predictions))
        print('\n===================')

def test_model2(model, cat):
    X = feature.values[cat]
    Y = clarity[cat]
    seed = 192 
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    model.fit(X_train)
    cc = Counter(y_train)
    print(cc)
    y_tune = model.predict(X_train)
    y_pred = model.predict(X_test)
    cc = Counter(y_tune)
    print(cc)
    if cc[1] > cc[0]:
        zero = 0
        one = 1
    else:
        zero = 1
        one = 0
    predictions = []
    ones = 0
    zeros = 0
    for item in y_pred:
        if item == zero:
            zeros += 1
            predictions.append(0)
        else:
            ones += 1
            predictions.append(1)
    
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('zero percentage %.4f' % (zero / (zero + one)))
    mse = list(map(lambda x,y: (x-y)*(x-y), y_test, predictions))
    print("RMSE: %.2f" % (np.sqrt(np.mean(mse))))
    print(confusion_matrix(y_test, predictions))
    print('\n===================')

if __name__ == '__main__':
    catall = list(range(row))
# ===== XGBoost Fashion =====
    print('XGBoost Fashion')
    model = xgbr(max_depth=7, learning_rate=0.5, objective='binary:logistic', nthread=2)
    test_model(model, cat)

# ===== XGBoost others =====
    print('XGBoost Others')
    model = xgbr(max_depth=7, learning_rate=0.5, objective='binary:logistic', nthread=2)
    test_model(model, cat2)

# ===== GMM Others =====
    print('GMM Others')
    model = gmm(n_components=2)
    test_model2(model, cat2)

# === Log Others =====
    print('Log Regression')
    model = lg()    
    test_model(model, cat)
