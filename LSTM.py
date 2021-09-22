# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:39:41 2021

@author: user
"""

print("測試")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from FinMind.data import DataLoader
dl = DataLoader()
# 下載台股股價資料
data = dl.taiwan_stock_daily(
    stock_id='2330', start_date='2018-01-01'
)

data.head()

from pathlib import Path

from flask import Flask, render_template, request
from loguru import logger
from pyecharts.charts import Page

import FinMind
from FinMind import plotting
from FinMind.data import DataLoader


def get_path():
     path = Path(FinMind.__file__)
     path = path.parent.joinpath("templates")
     return path


PATH = get_path()


def kline(data_loader: DataLoader, stock_id: str, start_date: str, end_date: str):
     stock_data = data_loader.taiwan_stock_daily(stock_id, start_date, end_date)
     stock_data = data_loader.feature.add_kline_institutional_investors(
          stock_data
     )
     stock_data = data_loader.feature.add_kline_margin_purchase_short_sale(
          stock_data
     )
     # 繪製k線圖
     kline_plot = plotting.kline(stock_data)
     return kline_plot
 
def bar(data_loader: DataLoader, stock_id: str, start_date: str, end_date: str):
     df = data_loader.taiwan_stock_month_revenue(
          stock_id=stock_id, start_date=start_date, end_date=end_date
     )
     df["labels"] = (
          df[["revenue_year", "revenue_month"]]
          .astype(str)
          .apply(lambda date: f"{date[0]}-{date[1]}M", axis=1)
     )
     df["series"] = df["revenue"].map(lambda value: round(value * 1e-8, 2))
     bar_plot = plotting.bar(
          labels=df["labels"],
          series=df["series"],
          title="月營收",
          yaxis_color="orange",
          y_axis_name="億",
     )
     return bar_plot
    
def line(data_loader: DataLoader, stock_id: str, start_date: str, end_date: str):
     df = data_loader.taiwan_stock_shareholding(
          stock_id=stock_id, start_date=start_date, end_date=end_date
     )
     df["series"] = df["ForeignInvestmentSharesRatio"].map(
          lambda value: round(value * 1e-2, 2)
     )
     df["labels"] = df["date"]
     line_plot = plotting.line(
          labels=df["labels"],
          series=df["series"],
          title="外資持股比例",
          yaxis_color="blue",
          y_axis_name="",
     )
     return line_plot
 

def pie(data_loader: DataLoader, stock_id: str, start_date: str, end_date: str):
     df = data_loader.taiwan_stock_holding_shares_per(
          stock_id=stock_id, start_date=start_date, end_date=end_date
     )
     df = df[df["date"] == max(df["date"])]
     df = df[df["HoldingSharesLevel"] != "total"]
     df["labels"] = df["HoldingSharesLevel"]
     df["series"] = df["percent"]
     pie_plot = plotting.pie(
          labels=df["labels"], series=df["series"], title="股權分散表"
     )
     return pie_plot
 

def dashboard(stock_id: str, start_date: str, end_date: str):
     data_loader = DataLoader()
     page = Page(layout=Page.SimplePageLayout)
     page.add(
          kline(data_loader, stock_id, start_date, end_date),
          bar(data_loader, stock_id, start_date, end_date),
          line(data_loader, stock_id, start_date, end_date),
          pie(data_loader, stock_id, start_date, end_date),
     )
     dashboard_html_path = str(PATH.joinpath("dashboard.html"))
     post_html_path = str(PATH.joinpath("post.html"))
     page.render(dashboard_html_path)
     post_html = open(post_html_path, "r", encoding="utf-8").read()
     dashboard_html = open(dashboard_html_path, "r", encoding="utf-8").read()
     html = post_html.replace("DASHBOARD", dashboard_html)
     with open(dashboard_html_path, "w", encoding="utf-8") as e:
          e.write(html)
         
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

keras.backend.clear_session()
regressor = Sequential()
regressor.add(LSTM(units = 100, input_shape = (X_train.shape[1], 1)))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.summary()

history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 16)
plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.plot( history.history["loss"])


dataset_total = pd.concat((train['open'], test['open']), axis = 0)
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


plt.plot(test['open'].values, color = 'black', label = 'Real TSMC Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TSMC Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
# plt.savefig('lstm_2330.png')