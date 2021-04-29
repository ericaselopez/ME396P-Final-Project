# main imports
%matplotlib inline
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
print(Reddit_data.head(5))


# calculate log returns
Reddit_data["log_returns"] = np.log(1 + Reddit_data["returns"])
print(Reddit_data.head(5))



# Let's Plot the log returns
plt.figure(2, figsize=(16,4))
plt.plot(Reddit_data.log_returns)
# we can see that it is much more stable now.



#lets observe the few item from X_red.
Reddit_data.dropna(inplace=True)
x_red = Reddit_data[["log_returns"]].values
print(x_red[:5])


