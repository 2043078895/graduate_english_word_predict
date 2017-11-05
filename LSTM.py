# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:25:05 2017

@author: Administrator
"""

from keras.layers import  LSTM
from keras.layers import  GRU
from keras.layers import  Dense
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import  regularizers
from keras import  optimizers
from keras.layers import  Flatten
from keras.models import  load_model
from keras.layers import  Masking
from keras.layers import Dropout
import os
class MYLSTM:
    def __init__(self,co_size,time_step,saved_path):
        self.save_path=saved_path
        if not os.path.exists(saved_path):
            self.LSTM_model=Sequential()
#            self.LSTM_model.add(Dense(10,input_shape=(co_size,),kernel_regularizer=regularizers.l2(0.00001),activation='relu'))
#            self.LSTM_model.add(Dropout(0.8))
#            self.LSTM_model.add(BatchNormalization())
#            self.LSTM_model.add(Dense(8,kernel_regularizer=regularizers.l2(0.00001),activation='relu'))
#            self.LSTM_model.add(Dropout(0.8))            
#            self.LSTM_model.add(Dense(5,kernel_regularizer=regularizers.l2(0.00001),activation='relu'))
#            self.LSTM_model.add(Dropout(0.8))
            
            self.LSTM_model.add(Masking(mask_value= -1,input_shape=(time_step,co_size)))
            self.LSTM_model.add(GRU(256,input_shape=(time_step,co_size),kernel_regularizer=regularizers.l2(0.01),dropout=0.8,return_sequences=True))#添加LSTM层                                
            self.LSTM_model.add(BatchNormalization())
            self.LSTM_model.add(GRU(128,kernel_regularizer=regularizers.l2(0.00001),return_sequences=True,dropout=0.5))
#            self.LSTM_model.add(BatchNormalization())
            self.LSTM_model.add(GRU(128,kernel_regularizer=regularizers.l2(0.00001),return_sequences=True,dropout=0.5))
            self.LSTM_model.add(BatchNormalization())
            self.LSTM_model.add(GRU(64,kernel_regularizer=regularizers.l2(0.00001),return_sequences=True,dropout=0.5))
            self.LSTM_model.add(BatchNormalization())
            self.LSTM_model.add(Dense(25,kernel_regularizer=regularizers.l2(0.00001),activation='relu'))
            self.LSTM_model.add(Dropout(0.5))
            self.LSTM_model.add(BatchNormalization())
            self.LSTM_model.add(GRU(32,kernel_regularizer=regularizers.l2(0.00001),return_sequences=True,dropout=0.5))
            self.LSTM_model.add(BatchNormalization())
            self.LSTM_model.add(Dense(10,kernel_regularizer=regularizers.l2(0.00001),activation='relu'))
            self.LSTM_model.add(Dropout(0.5))
            self.LSTM_model.add(BatchNormalization())
            self.LSTM_model.add(GRU(1,kernel_regularizer=regularizers.l2(0.00001),dropout=0.5,activation='sigmoid'))
#            self.LSTM_model.add(Flatten())                
#            self.LSTM_model.add(Dense(10,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
##            self.LSTM_model.add(Flatten())  
#            self.LSTM_model.add(Dense(1,activation='sigmoid'))
        else:
            print('laoded the model!')
            self.LSTM_model=load_model(saved_path)            
        self.optim=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.LSTM_model.compile(loss='binary_crossentropy', optimizer=self.optim,metrics=['acc']) #编译模型
            
    def fit(self,train_set,target_set,nb_epoch = 10, batch_size = 500):
        self.LSTM_model.fit(train_set,target_set,batch_size=batch_size, epochs=nb_epoch)
    def predict(self,predict_X):
        return self.LSTM_model.predict(predict_X)
    def save_model(self):
        self.LSTM_model.save(self.save_path)
        print('Saved the model successfully!')

        
        