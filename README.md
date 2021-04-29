# Predicting TSLA Stock Behavior using a Neural Network
Final Project codebase for ME396P Group 9 (The AIs of Texas)

Step 1: Download the training data

Under the Training Data branch, you can find the Reddit data (april20toapril21_redditdata.csv) and the TSLA stock price data (TSLAstockdata.csv). We have also included the Google Trends data (googletrendsdata.csv), though we did not use it in our final version of our model. This data...

If you would like to pull data from Reddit yourself, download the searchreddit.py file. You will need to create a client ID, client secret, and user agent yourself by registering an application on Reddit. Go here for the instructions on how to do so (https://praw.readthedocs.io/en/latest/getting_started/quick_start.html) and here to register your app (https://www.reddit.com/prefs/apps/).



Step 2. Build and train your model

Once you get the Reddit data and Stock data for TSLA from the training data folder, you can use LSTM_Prediction.py code to run it and traing the LSTM and get your predictions. You can change the Epoch, number of nodes in LSTM layer, number of Training and Testing percentage, etc. and see the results.

You can also choose to have Reddit data as your training data or closing price as your training data (X values). Google Trend data leads to very similar results to Reddit data. If you want to test that as well, we provided the data in the training data folder.

If you want to do this prediction without transformation of the data, you need to comment the codes for transformation and inverse transformation.
Also, if you want to have both Reddit and Stock as your data, you can combine them easily and train the model.

The loss function is chosed to be mean square error and you can change it to your favorite loss function if you want.

At the end of the code, we plot the true values and predicted values so that we compare the accuracy of the prediction visually.
