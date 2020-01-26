#Train On Multiple Lag Timesteps Example
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
# convert series to supervised learning

#   data: Sequence of observations as a list or 2D NumPy array. Required.
#   n_in: Number of lag observations as input (X). Values may be between [1..len(data)] Optional. Defaults to 1.
#   n_out: Number of observations as output (y). Values may be between [0..len(data)-1]. Optional. Defaults to 1.
#   dropnan: Boolean whether or not to drop rows with NaN values. Optional. Defaults to True.
#   return: Pandas DataFrame of series framed for supervised learning.
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('SampledPM2_5.csv', header=0, index_col=0)
values = dataset.values
print(dataset.head())

# integer encode direction
#encoder = LabelEncoder()
#values[:,3] = encoder.fit_transform(values[:,3])

values = values.astype('float32')   # ensure all data is float

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print('Values shape:',values.shape)

values3 = scaler.inverse_transform(scaled)
values4 = values3[:,0]

# specify the number of lag hours
n_hours = 24
n_features = 3

# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1) #Reframed dataset for traning
 
# split into train and test sets
values = reframed.values
n_train_hours = 30 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features] # Small y means single-variable prediction
test_X, test_y = test[:, :n_obs], test[:, -n_features]

train_tX = train_X
train_ty = train_y

print('---------------------------------------')
print('Shape of reframed data:',reframed.shape)
print('Shape of train_X data :',train_X.shape)
print('Shape of train_y data :',train_y.shape)
print('Remaining hours:',len(dataset.index)-len(train_y)-len(test_y))


#count_row = df.shape[0]  # gives number of row count
#count_col = df.shape[1]  # gives number of col count
#count_col = df.shape[2]  # 


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

print('---------------------------------------')
print('train_X, train_y, test_X, test_y:')
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print('---------------------------------------')

# design network
# 50 neurons in the first hidden layer
# 1 neuron in the output layer for predicting pollution.
# The input shape will be 1 time step with 8 features.
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=60, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
pyplot.figure(1)
pyplot.plot(history.history['loss'], label='Training loss')
pyplot.plot(history.history['val_loss'], label='Testing loss')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

# invert scaling for forecast data
inv_yhat = concatenate((yhat, test_X[:, -2:]), axis=1)  # Combine for enough dimension for inverse transform
inv_yhat = scaler.inverse_transform(inv_yhat)           # restore the scaled data
inv_yhat = inv_yhat[:,0]                                # choose only predicted data

# invert scaling for actual data (test_y)
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -2:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0] # Inverted y

# invert scaling for actual data (train_y)
train_ty = train_y.reshape((len(train_ty), 1))
inv_ty = concatenate((train_ty, train_tX[:, -2:]), axis=1)
inv_ty = scaler.inverse_transform(inv_ty)
inv_ty = inv_ty[:,0] # Inverted ty

pyplot.figure(2)
#pyplot.plot(yhat)
pyplot.plot(inv_y)
pyplot.plot(inv_yhat)
pyplot.legend(['Test_Y', 'Predic_Y'])
pyplot.ylabel("ug/m3")
pyplot.xlabel("Hour")
pyplot.show()

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

