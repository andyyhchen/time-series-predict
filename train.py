import utils
import numpy as np 
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

#load data


def create_dataset(row, look_back=1):
	dataX, dataY = [], []
	for i in range(len(row) - look_back -1):
		a = row[i:(i+look_back)]
		dataX.append(a)
		dataY.append(row[i+look_back].reshape(1,))

	return np.array(dataX), np.array(dataY)

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def predict_full_sequence(model, row, look_back=1, timestemps=1):
	predicted = row[:look_back]
	for i in range(len(row) - look_back):
		tmp = model.predict(np.array(predicted[-look_back:]).reshape(1, timestemps, look_back))
		predicted = np.append(predicted, tmp)
	return predicted




def run(row, look_back=14):

	train_len = int(len(row)*0.6)
	test_len = len(row) - train_len

	trainX, trainY = create_dataset(row[:train_len], look_back = look_back)

	print (trainY.shape)
	
	timestemps=1
	data_dim = trainX.shape[1]

	model = Sequential()
	model.add(LSTM(32, return_sequences=True, input_shape=(timestemps, data_dim)))
	model.add(LSTM(32, return_sequences=True))
	model.add(LSTM(32))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')

	trainX = trainX.reshape(trainX.shape[0], timestemps, trainX.shape[1])
	trainY = trainY.reshape(trainY.shape[0], 1)

	model.fit(trainX, trainY, batch_size=32, epochs=100, validation_split=0.05)

	return model


if __name__=='__main__':

	data = utils.read_data('/home/andyisman/Desktop/TimeSeriesPredictionRnn/AnswerClean_haveNA_.csv')
	look_back = 14
	row = data[5]
	train_len = int(len(row)*0.8)
	trow = row[train_len:]
	model = run(row, look_back)
	predicted = predict_full_sequence(model, trow, look_back=look_back, timestemps=1)
	print(predicted.shape, trow.shape)
	plot_results(predicted, trow)




