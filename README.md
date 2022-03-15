# ESSAY
這邊是放著我論文的Demo，內容是股市分析。

#### 大綱



先把lstm做出來:

```python

# lstmmodel.add(tf.keras.layers.LSTM(50,input_shape = (lstmlookback, lstmn_features), return_sequences=True))
lstmmodel.add(tf.keras.layers.LSTM(512, activation='relu',input_shape = (lstmlookback, lstmn_features), return_sequences=True))
lstmmodel.add(tf.keras.layers.LSTM(256, activation='relu'))
lstmmodel.add(tf.keras.layers.Dense(1))
# lstmmodel.add(tf.keras.layers.Activation('softmax'))
# print(lstmmodel.summary())


lstmmodel.compile(loss = 'mse', optimizer = 'adam')
print(lstmmodel.summary())
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')

start = time()
print("start:",0)
history = lstmmodel.fit(lstmx_train,lstmy_train, epochs = 1000, batch_size=128, 
          validation_data=(lstmx_val,lstmy_val),verbose = 1, 
          shuffle = False, callbacks=[earlystop])
print("end:",time()-start)

```

