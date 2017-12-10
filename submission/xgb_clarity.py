import pandas as pd
import numpy as np
from xgboost import XGBClassifier as xgbc
from xgboost import XGBRegressor as xgbr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# load data
infile = pd.read_csv('../data/Clean_Data_Frame.csv')
infile2 = pd.read_json('../data/Dataframe_Of_Everything.json', lines=True)
row, _ = infile.shape
cat = []
for i in range(row):
    dfl = infile.iloc[i]
    if str(dfl['category_1']) == 'Fashion' or str(dfl['category_1']) == 'Watches Sunglasses Jewellery':
    # if str(dfl['category_1']) != 'Fashion' and str(dfl['category_1']) != 'Watches Sunglasses Jewellery':
        cat.append(i)
clarity = infile['clarity']
feature = pd.read_pickle('final_features.bin')
# split data into train and test sets
X = feature.values[cat]
Y = clarity[cat]
seed = 192 
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = xgbr(max_depth=12, learning_rate=0.8, objective='binary:logistic', nthread=2)
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
# y_pred = model.predict(X_train)
# predictions = [round(value) for value in y_pred]
for i in np.arange(0.4, 1, 0.1):
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
# evaluate predictions
