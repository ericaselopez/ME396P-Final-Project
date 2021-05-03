# main imports
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


# several imports from Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean



# reading stock data (if you are using the Google Colab)
from google.colab import files 
import pandas as pd
import io


# For data transformation
from sklearn.preprocessing import MinMaxScaler


# For LSTM method:
from keras.models import Sequential
from keras.layers import LSTM, Dense


from sklearn.metrics import mean_squared_error


# If you are using Google Colab, change it to True
Colab = False

# If you want to train the data based on reddit data, change it True. 
Reddit = False


# Read reddit data
if Colab:
  Reddit_data = pd.read_csv(io.StringIO(files.upload()['april20toapril21_redditdata.csv'].decode('utf-8')))
else:
  Reddit_data = pd.read_csv('april20toapril21_redditdata.csv')


Reddit_data.set_index("Date", drop=True, inplace=True)
Reddit_data = Reddit_data[["TSLA_Count"]]

# For observations only!
print(Reddit_data.describe())
print(Reddit_data.info())
print(Reddit_data.values.shape)


# plot Reddit data:
plt.figure(1, figsize=(16,6))
_ = plt.plot(Reddit_data.TSLA_Count)
# We can see that the data is not "stationary". In order to make it more stable, we need to transform the data into a form that is more stable.


# Create a new data frame with TSLA Count and percentage change
Reddit_data["returns"] = Reddit_data.TSLA_Count.pct_change()

# calculate log returns
Reddit_data["log_returns"] = np.log(1 + Reddit_data["returns"])


# Let's Plot the log returns
plt.figure(2, figsize=(16,4))
plt.plot(Reddit_data.log_returns)
# we can see that it is much more stable now.



#lets observe the few item from X_red.
Reddit_data.dropna(inplace=True)
x_red = Reddit_data[["log_returns"]].values
print("reddit data: transformed into log returns")
print(x_red[:5])


if Colab:
  Stock_data = pd.read_csv(io.StringIO(files.upload()['TSLA_apr20_apr21.csv'].decode('utf-8')))
else:
  Stock_data = pd.read_csv('TSLA_apr20_apr21.csv')

#only using closing price
Stock_data.set_index("Date", drop=True, inplace=True)
Stock_data = Stock_data[["Close", "TSLA_Count"]]
print("describe the stock data:")
print(Stock_data.describe())


# Plot the Closing price
plt.figure(1, figsize=(16,6))
_ = plt.plot(Stock_data.Close)


# Create a new data frame with closing price and percentage change
Stock_data["returns"] = Stock_data.Close.pct_change()

# calculate log returns
Stock_data["log_returns"] = np.log(1 + Stock_data["returns"])

# let's look at the first 5 data points
print("stock data:")
print(Stock_data.head(5))



# Let's Plot the log returns
plt.figure(1, figsize=(16,4))
plt.plot(Stock_data.log_returns)



Stock_data.dropna(inplace=True)
x_stock = Stock_data[["Close", "log_returns"]].values


# Data transformation between zero and one:
scaler_raw = MinMaxScaler(feature_range=(0,1)).fit(x_stock)
X_stock_scaled = scaler_raw.transform(x_stock)

scaler_red = MinMaxScaler(feature_range=(0,1)).fit(x_red)
X_red_scaled = scaler_red.transform(x_red)


# Here is the Y value for the 
y = [x[0] for x in X_stock_scaled]

# 80 percent for training and 20 percent for testing.
split = int(len(X_stock_scaled)*0.8)
if not Reddit:
  # First Option for Training: Stock based on Stock
  X_train = X_stock_scaled[:split]
  X_test = X_stock_scaled[split:len(X_stock_scaled)]

  Y_train = y[:split]
  Y_test = y[split:len(y)]

else:
  # Second Option for Training: Stock based on Reddit

  X_train = X_red_scaled[:split]
  X_test = X_red_scaled[split:len(X_red_scaled)]

  Y_train = y[:split]
  Y_test = y[split:len(y)]


# To check if dimensions are consistent
assert len(X_train) == len(Y_train)
assert len(X_test) == len(Y_test)



# looking n days in the past
# if n=3, it means that we are looking for day 1,2,3 and predicting day 4.

n=3
Xtrain = []
Ytrain = []
Xtest = []
Ytest = []

for i in range(n, len(X_train)):
  Xtrain.append(X_train[i-n : i, : X_train.shape[1]])
  Ytrain.append(Y_train[i])


for i in range(n, len(X_test)):
  Xtest.append(X_test[i-n : i , : X_test.shape[1]])
  Ytest.append(Y_test[i])


# inverse transform to see the actual original values
val = np.array(Ytrain[0])
val = np.c_[val, np.zeros(val.shape)]

# for stock based on stock
scaler_raw.inverse_transform(val)


# LSTM METHOD
# we need to know the number of observations, time steps, and features in each steps

Xtrain, Ytrain = (np.array(Xtrain), np.array(Ytrain))
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))

Xtest, Ytest = (np.array(Xtest), np.array(Ytest))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))



model = Sequential()
model.add(LSTM(512, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(Xtrain, Ytrain, epochs=75, validation_data = (Xtest,Ytest), batch_size=16, verbose=1)

print("model summary")
print(model.summary())


# Predictions:

trainPredict = model.predict(Xtrain)
testPredict = model.predict(Xtest)

trainPredict = np.c_[trainPredict, np.zeros(trainPredict.shape)]
testPredict = np.c_[testPredict, np.zeros(testPredict.shape)]


# inverse transformed predictions:
trainPredict = scaler_raw.inverse_transform(trainPredict)
trainPredict = [x[0] for x in trainPredict]

testPredict = scaler_raw.inverse_transform(testPredict)
testPredict = [x[0] for x in testPredict]


# Calculate Root mean square Error.
trainScore = mean_squared_error([x[0][0] for x in Xtrain], trainPredict, squared=False)
print("Train Score: %.2f RMSE" % (trainScore))

testScore = mean_squared_error([x[0][0] for x in Xtest], testPredict, squared=False)
print("Test Score: %.2f RMSE" % (testScore))


# True data:
orig_val = []
for i in range(len(Stock_data)):
  orig_val.append(Stock_data["Close"][i])

# Predicted data
predicted = trainPredict + testPredict


# Plot the results
plt.figure(1, figsize=(16,8))
plt.plot(orig_val[n:-n])
plt.plot(predicted)
plt.ylabel("TSLA Closing Price")
plt.xlabel("Week Days")
plt.legend(["True", "Predicted"])
plt.show()
