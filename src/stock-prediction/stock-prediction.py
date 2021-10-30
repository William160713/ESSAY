import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from time import time
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from time import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
#顯示中文

from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False



lstmdf = pd.read_csv("../../data/MSFT_complete.csv",parse_dates=True,index_col="Date")
#df = df[1563:]
del lstmdf["Unnamed: 0"]
lstmdf.drop(columns=['Volume','Adj Close'],inplace=True)
lstmdf.columns = [['open', 'high', 'low','close',]]
print("---dataframe head---")
print(lstmdf.head())

print("--scaling data---")
lstmddata = sc.fit_transform(lstmdf)

lstmdtrain_ind = int(0.6*len(lstmdf))
lstmdval_ind = lstmdtrain_ind + int(0.2*len(lstmdf))

lstmtrain = lstmddata[:lstmdtrain_ind]
lstmval = lstmddata[lstmdtrain_ind:lstmdval_ind]
lstmtest = lstmddata[lstmdval_ind:]

print("--shapes--")
print("lstmtrain,lstmtest,lstmval",lstmtrain.shape, lstmtest.shape, lstmval.shape)


lstmnxtrain,lstmytrain,lstmxval,lstmyval,lstmxtest,lstmytest = lstmtrain[:,:4],lstmtrain[:,3],lstmval[:,:4],lstmval[:,3],lstmtest[:,:4],lstmtest[:,3]



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
plt.plot(loss['loss'], label='loss')
plt.plot(loss['val_loss'], label='val_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.savefig("./plots/loss-vadercase6.jpg")
plt.legend()
plt.show()
lstmy_pred = lstmmodel.predict(lstmx_test)
print("r2_score:",r2_score(lstmy_pred,lstmy_test))


plt.figure(figsize=(20,10))
plt.plot( lstmy_test, '.-', color='red', label='真實股價走向', alpha=0.5)
plt.plot( lstmy_pred, '.-', color='black', label='LSTM', alpha=1)
#---------------
plt.title('股票預測')
plt.xlabel('日期')
plt.ylabel('收盤價')
plt.legend() #圖例
#---------------
plt.savefig("./plots/result-vadercase6.jpg")
plt.show()





