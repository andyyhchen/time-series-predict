import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def read_data(path):

	'''
	Input: csv data path /home/andyisman/Desktop/TimeSeriesPredictionRnn/AnswerClean_haveNA_.csv
	Output: 2d numpy array of data
	'''

	data = pd.read_csv(path)
	data = data.iloc[:, 1:]
	data = data.interpolate(method='linear', axis=1).bfill()
	data = np.array(data)[:,1:]


	return data

def normalize_data(row):
	scalar = MinMaxScaler()
	scalar.fit(row.reshape(-1,1))
	trow = scalar.transform(row1.reshape(-1,1)).flatten()

	return trow, scalar

