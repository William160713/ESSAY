import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
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

#MinMaxScaler

#-------------------------------------------------------------------------
"""
在MinMaxScaler中是給定了一個明確的最大值與最小值。
每個特徵中的最小值變成了0，最大值變成了1。數據會縮放到到[0,1]之間。
"""

#-------------------------------------------------------------------------

#顯示中文

#-------------------------------------------------------------------------
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
#-------------------------------------------------------------------------

#資料預處理
#以下為lstm
lstmdf = pd.read_csv("../../data/MSFT_complete.csv",parse_dates=True,index_col="Date")
#df = df[1563:]
del lstmdf["Unnamed: 0"]
lstmdf.drop(columns=['Volume','Adj Close'],inplace=True)
lstmdf.columns = [['open', 'high', 'low','close',]]
print("---dataframe head---")
print(lstmdf.head())


#以下為lstm+sentiment
df = pd.read_csv("../../data/microsoft_verified_users/verified_MSFT-vader.csv",parse_dates=True,index_col="Date")
#-------------------------------------------------------------------------

 #df = df[1563:]
print(df.corr()['Close'])
del df["Unnamed: 0"]
del df["Unnamed: 0.1"]
#刪除直接使用.drop()，裡面放入要刪除的labels
df.drop(columns=['Volume','Adj Close'],inplace=True)
df.columns = [['open', 'high', 'low','close','sentiment']]
print("---lstm dataframe head---")
print(df.head())



#以下為lstm 縮放數據的部分
#-------------------------------------------------------------------------

print("--lsrm scaling data---")
lstmddata = sc.fit_transform(lstmdf)

lstmdtrain_ind = int(0.6*len(lstmdf))
lstmdval_ind = lstmdtrain_ind + int(0.2*len(lstmdf))

lstmtrain = lstmddata[:lstmdtrain_ind]
lstmval = lstmddata[lstmdtrain_ind:lstmdval_ind]
lstmtest = lstmddata[lstmdval_ind:]

print("--lstm shapes--")
print("lstmtrain,lstmtest,lstmval",lstmtrain.shape, lstmtest.shape, lstmval.shape)

lstmnxtrain,lstmytrain,lstmxval,lstmyval,lstmxtest,lstmytest = lstmtrain[:,:4],lstmtrain[:,3],lstmval[:,:4],lstmval[:,3],lstmtest[:,:4],lstmtest[:,3]


#-------------------------------------------------------------------------
#以下為lstm+sentiment的部分
print("--scaling data 縮放數據---")
data = sc.fit_transform(df) 
train_ind = int(0.6*len(df))
val_ind = train_ind + int(0.2*len(df))

train = data[:train_ind]
val = data[train_ind:val_ind]
test = data[val_ind:]

print("--shapes print訓練 測試 驗證的陣列大小--")
print("train,test,val",train.shape, test.shape, val.shape)


xtrain,ytrain,xval,yval,xtest,ytest = train[:,:5],train[:,3],val[:,:5],val[:,3],test[:,:5],test[:,3]
print("-------------------------------------------------------------------------")
#-------------------------------------------------------------------------
#以下為lstm時間步的部分

lstmlookback = 60
lstmn_features = 4
lstmtrain_len = len(lstmnxtrain) - lstmlookback
test_len = len(lstmxtest) - lstmlookback
lstmval_len = len(lstmxval) - lstmlookback

lstmx_train = np.zeros((lstmtrain_len, lstmlookback, lstmn_features))
lstmy_train = np.zeros((lstmtrain_len))
for i in range(lstmtrain_len):
    lstmytemp = i+lstmlookback
    lstmx_train[i] = lstmnxtrain[i:lstmytemp]
    lstmy_train[i] = lstmytrain[lstmytemp]
print("lstmx_train", lstmx_train.shape)
print("lstmy_train", lstmy_train.shape)

lstmx_test = np.zeros((test_len, lstmlookback, lstmn_features))
lstmy_test = np.zeros((test_len))
for i in range(test_len):
    lstmytemp = i+lstmlookback
    lstmx_test[i] = lstmxtest[i:lstmytemp]
    lstmy_test[i] = lstmytest[lstmytemp]
print("lstmx_test", lstmx_test.shape)
print("lstmy_test", lstmy_test.shape)

lstmx_val = np.zeros((lstmval_len, lstmlookback, lstmn_features))
lstmy_val = np.zeros((lstmval_len))
for i in range(lstmval_len):
    lstmytemp = i+lstmlookback
    lstmx_val[i] = lstmxval[i:lstmytemp]
    lstmy_val[i] = lstmyval[lstmytemp]
print("lstmx_val", lstmx_val.shape)
print("lstmy_val", lstmy_val.shape)

print("-------------------------------------------------------------------------")
#-------------------------------------------------------------------------
#以下為lstm+sentiment的部分
#lookback 個時間步
#n_features 每個steps裡5個時間步

lookback = 60
n_features = 5
train_len = len(xtrain) - lookback
test_len = len(xtest) - lookback
val_len = len(xval) - lookback

#返回一個給定形狀和類型的用0填充的數組
x_train = np.zeros((train_len, lookback, n_features))
y_train = np.zeros((train_len))
for i in range(train_len):
    ytemp = i+lookback
    x_train[i] = xtrain[i:ytemp]
    y_train[i] = ytrain[ytemp]
print("x_train", x_train.shape)
print("y_train", y_train.shape)

x_test = np.zeros((test_len, lookback, n_features))
y_test = np.zeros((test_len))
for i in range(test_len):
    ytemp = i+lookback
    x_test[i] = xtest[i:ytemp]
    y_test[i] = ytest[ytemp]
print("x_test", x_test.shape)
print("y_test", y_test.shape)
print("-------------------------------------------------------------------------")
#-------------------------------------------------------------------------

#處理訓練測試驗證的資料
x_val = np.zeros((val_len, lookback, n_features))
y_val = np.zeros((val_len))
for i in range(val_len):
    ytemp = i+lookback
    x_val[i] = xval[i:ytemp]
    y_val[i] = yval[ytemp]
print("x_val", x_val.shape)
print("y_val", y_val.shape)
print("-------------------------------------------------------------------------")
#-------------------------------------------------------------------------
#lstm模型
lstmmodel = Sequential() 
lstmmodel.add(LSTM(600,input_shape = (lstmlookback, lstmn_features), return_sequences=True))
lstmmodel.add(LSTM(700))
lstmmodel.add(Dropout(0.15))
lstmmodel.add(Dense(1))
print(lstmmodel.summary())

lstmmodel.compile(loss = 'mse', optimizer = 'adam')
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')

start = time()
print("start:",0)
history = lstmmodel.fit(lstmx_train,lstmy_train, epochs = 10, batch_size=30, 
          validation_data=(lstmx_val,lstmy_val),verbose = 1, 
          shuffle = False, callbacks=[earlystop])
print("end:",time()-start)

lstmmodel.save("./models/model-vadercase6.h5")
loss = history.history
plt.plot(loss['loss'], label='lstmloss')
plt.plot(loss['val_loss'], label='lstmval_loss')
plt.savefig("./plots/loss-vadercase6.jpg")
plt.legend()
plt.show()

lstmy_pred = lstmmodel.predict(lstmx_test)
print("r2_score:",r2_score(lstmy_pred,lstmy_test))
print("-------------------------------------------------------------------------")




#-------------------------------------------------------------------------
#lstm+sentiment 模型
model = Sequential()  
model.add(LSTM(600,input_shape = (lookback, n_features), return_sequences=True))
model.add(LSTM(700))
model.add(Dropout(0.15))
model.add(Dense(1))
print(model.summary())
print("-------------------------------------------------------------------------")
model.compile(loss = 'mse', optimizer = 'adam')


"""
EarlyStopping 顧名思義就是提前中止訓練，一般來說會在下列情況下停止訓練：

出現 Overfitting 的現象。
模型指標沒有明顯改進。例如：loss 不降、acc 不升…等。
模型收斂不了或收斂過慢。

這邊調用驗證集monitor='val_loss'
min_delta:增大或減少的閥值
patience: 多少個 epoch 內監控的數據都沒有出現改善?
verbose:有 0 或 1 兩種設置。 0 是 silent 不會輸出任何的訊息， 1 的話會輸出一些 debug 用的訊息。
mode: 有 auto, min 和 max 三種設置選擇。用來設定監控的數據的改善方向，如過希望你的監控的數據是越大越好，則設置為 max，如：acc；反之，若希望數據越小越好，則設定 min，如：loss。
"""
"""
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')

start = time()
print("start:",0)
history = model.fit(x_train,y_train, epochs = 1, batch_size=30, 
          validation_data=(x_val,y_val),verbose = 1, 
          shuffle = False, callbacks=[earlystop])
print("endtime:",time()-start)
"""
print("-------------------------------------------------------------------------")
#<<<<<<< HEAD
"""
model.save("./models/model_vader7.h5")
loss = history.history
plt.plot(loss['loss'], label='loss')
plt.plot(loss['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend() #圖例

plt.savefig("./plots/loss_vader7.jpg")
"""
#=======
"""
model.save("./models/model_vader6.h5")
loss = history.history
plt.plot(loss['loss'], color='red', label='lstm+sentiment_loss')
plt.plot(loss['val_loss'], color='blue', label='lstm+sentiment_val_loss')
plt.savefig("./plots/loss_vader6.jpg")
#>>>>>>> 7bf5ab529d185e0cc646bfcf14a8abb1ed7a4042
plt.legend() 
plt.show()

model = load_model("./models/model_vader1.h5")
y_pred = model.predict(x_test)
print(model.summary())
#-------------------------------------------------------------------------
#這邊開始繪圖，把lstm跟lstm+sentiment的指數合併在一起

plt.figure(figsize=(20,10))
plt.plot( y_test, '.-', color='red', label='真實股價走向', alpha=0.5)
plt.plot( y_pred, '.-', color='blue', label='LSTM+sentiment', alpha=1)
#這邊要再多一條黃色的表達單一的LSTM
plt.plot( lstmy_pred, '.-', color='black', label='LSTM', alpha=1)

#<<<<<<< HEAD
plt.savefig("./plots/result_vader7.jpg")
#=======
plt.savefig("./plots/result_vader6.jpg")
#>>>>>>> 7bf5ab529d185e0cc646bfcf14a8abb1ed7a4042
#---------------
plt.title('股票預測')
plt.xlabel('日期')
plt.ylabel('收盤價')
plt.legend() #圖例
#---------------
plt.show()

print("lstmr2_score:",r2_score(y_pred,y_test))
"""




