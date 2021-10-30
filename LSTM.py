# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:39:41 2021

@author: user
"""

print("實驗開始")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from FinMind.data import DataLoader
dl = DataLoader()
# 下載台股股價資料
data = dl.taiwan_stock_daily(
    stock_id='0050', start_date='2018-01-01'
)

data.head()

from pathlib import Path

from flask import Flask, render_template, request
from loguru import logger
from pyecharts.charts import Page

import FinMind
from FinMind import plotting
from FinMind.data import DataLoader

     
#切分Test集
test = data[data.date>'2019-09-01']
train = data[:len(data)-len(test)]
#只要open high
train_set = train['open']
test_set = test['open']

from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range = (0, 1))
#需將資料做reshape的動作，使其shape為(資料長度,1) 
train_set= train_set.values.reshape(-1,1)
training_set_scaled = sc.fit_transform(train_set)


X_train = [] 
y_train = []
for i in range(10,len(train_set)):
    X_train.append(training_set_scaled[i-10:i-1, 0]) 
    y_train.append(training_set_scaled[i, 0]) 
X_train, y_train = np.array(X_train), np.array(y_train) 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print('第i天之前的股價')
print(X_train[0])
print('第i天的股價')
print(y_train[0])


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout,BatchNormalization  
from keras import optimizers
from keras.layers import Activation, Dense      


keras.backend.clear_session()
regressor = Sequential()
regressor.add(LSTM(units = 200, input_shape = (X_train.shape[1], 1)))

#加入激活函數
#regressor.add(Activation('tanh'))
regressor.add(Activation('softsign'))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error' ,metrics=['accuracy'])

regressor.summary()



history = regressor.fit(X_train, y_train, epochs = 1000, batch_size = 16)


plt.subplot(1, 2, 1) #讓圖表同時排列方便瀏覽
plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.plot( history.history["loss"])



dataset_total = pd.concat((train['close'], test['close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test) - 10:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(10, len(inputs)):
    X_test.append(inputs[i-10:i-1, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted_stock_price = regressor.predict(X_test)
#使用sc的 inverse_transform將股價轉為歸一化前
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.subplot(1, 2, 2) 
plt.plot(test['open'].values, color = 'black', label = 'Real TSMC Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TSMC Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()


plt.savefig('C:/Users/user/OneDrive - 國立宜蘭大學/文件/ESSAY/exresult/1000epoch-softsignadamstockresult-0050 .png')
plt.show()
plt.close()
print("實驗結束")


# plt.savefig('lstm_2330.png') 