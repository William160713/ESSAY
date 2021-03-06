# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:39:41 2021

@author: user
"""

print("實驗開始")
#開始嘗試加入心情指數
#以下為原本就有import的library--------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
#以下開始加入使用情感指數需要用到的library---------------------------------------
import matplotlib
import tensorflow as tf
from time import time
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from time import time
from keras.callbacks import EarlyStopping
#決定係數:越靠近0模型越差，越靠近1越好
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
#-----------------------------------------------------------------------------

#顯示中文
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
#-----------------------------------------------------------------------------


from FinMind.data import DataLoader
api = DataLoader()
# 下載台股加權資料

data = df = api.taiwan_stock_total_return_index(
    index_id="TAIEX",
    start_date='2011-01-01',
    end_date='2021-01-01'
)


#print(df)
#------------------------------------------------------------------------------
#2021/11/29 新增:現在把情感指數給讀進來
m2324_df = pd.read_csv("C:/Users/user/OneDrive - 國立宜蘭大學/文件/ESSAY/essaydb/M2324_result.csv",parse_dates=True)
#print(m2324_df.corr()['sentiment'])
#head 當中可以填入要找的行數讓python 返回前n行出來
#print(m2324_df.head())
#print(m2324_df['date'])

#讀取情感指數OK

#-----------------------------------------------------------------------------
from pathlib import Path

from flask import Flask, render_template, request
from loguru import logger
from pyecharts.charts import Page

import FinMind
from FinMind import plotting
from FinMind.data import DataLoader

     
#切分Test集
test = data[data.date>'2019-01-02']
train = data[:len(data)-len(test)]


#只要價格
train_set = train['price']
test_set = test['price']

#額外資訊: 增加情感指數


m2324_train_sentiment = m2324_df['sentiment']
print( "要使用的情感指數:", m2324_train_sentiment)


#m2324_train_set= m2324_train_sentiment.values.reshape(-1,1)
#m2324_training_set_scaled = sc.fit_transform(m2324_train_set)
#print("m2324_training_set_scaled的值 ",m2324_training_set_scaled )



#原始的股票處理----------------------------
from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range = (0, 1))
#需將資料做reshape的動作，使其shape為(資料長度,1) 
train_set= train_set.values.reshape(-1,1)
training_set_scaled = sc.fit_transform(train_set)
#----------------------------

"""
這裡需要特別解釋一下，for迴圈中的10代表著,我們將10天的資料視為一句話(一個序列的意思)，
因為序列型資料需要這樣子去呈現才行(X_train),而y_train則是用第i天前的資料來預測第i天的股價的意思。

"""


X_train = [] 
y_train = []
for i in range(10,len(train_set)):
    X_train.append(training_set_scaled[i-10:i-1, 0]) 
    y_train.append(training_set_scaled[i, 0])
        
X_train, y_train = np.array(X_train), np.array(y_train) 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))




#print('第i天之前的股價')
#print(X_train[0])
#print('第i天的股價')
#print(y_train[0])

#----------------------------------------------------------------



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

history = regressor.fit(X_train, y_train, epochs = 10, batch_size = 16)


plt.subplot(1, 2, 1) #讓圖表同時排列方便瀏覽
plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.plot( history.history["loss"])



dataset_total = pd.concat((train['price'], test['price']), axis = 0)
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
plt.plot(test['price'].values, color = 'black', label = '加權報酬指數真實值')
plt.plot(predicted_stock_price, color = 'green', label = '加權報酬指數預測值')
plt.title('加權報酬指數預測')
plt.xlabel('經過時間(days)')
plt.ylabel('加權報酬指數')
plt.legend()


plt.savefig('C:/Users/user/OneDrive - 國立宜蘭大學/文件/ESSAY/exresult/TAIEX_Total_Return_Index_Stock_Price-10epoch-softsignadamstockresult .png')
plt.show()
#plt.close()
print("實驗結束")


# plt.savefig('lstm_2330.png') 
