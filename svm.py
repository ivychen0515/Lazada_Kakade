import pandas as pd
import numpy as np
def getDataframe(filePath):
	dataframe = pd.read_csv(filePath)
	concise = dataframe['concise']
	clarity = dataframe['clarity']
	x = dataframe.drop(['concise', 'clarity', 'index'], axis=1)
	return x, clarity, concise

if __name__ == '__main__':
	path = './Clean_Data_Frame.csv'
	x, clarity, concise = getDataframe(path)
	print(x[0])

