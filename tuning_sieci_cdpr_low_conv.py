import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.layers import Input,BatchNormalization,GlobalMaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError,MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import ReduceLROnPlateau,EarlyStopping
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae

import tensorflow as tf
import numpy as np
from sieci_na_szybko import siec_konwolucyjna
import gc
import os
class TerminateOnNaN(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if logs.get('loss') is not None and np.isnan(logs.get('loss')):
            print("Wykryto NaN w lossie, przerywanie treningu.")
            self.model.stop_training = True


def create_shiftedDF(df,lags):
    indexes = list(df.index)[lags:]
    a = df.values
    vals = []
    for i in range(len(df)-lags):
        vals.append(a[i:lags+i])
    vals = pd.DataFrame(vals,indexes)
    return vals 
def recursive_array(df, lags):
    a = df.values
    x = []
    y=[]
    for i in range(len(df)-lags):
        x.append(a[i:lags+i])
        y.append(a[lags+i,0])
    return np.array(x),np.array(y)
warnings.filterwarnings("ignore")
start_date = "2022-01-01"
end_date = "2024-09-17"
ticker = "CDR.WA"

data = yf.download(ticker,start_date,end_date)
data.head()
rec = data[['High','Low','Open','Close']]
rec_h = rec.drop(columns = 'Low')
rec_l = rec.drop(columns = 'High')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # wskaźnik, na podstawie którego zmieniamy learning rate
                              factor=0.1,  # współczynnik redukcji (np. zmniejsza o 20%)
                              patience=1,  # liczba epok bez poprawy, po której następuje zmniejszenie learning rate
                              min_lr=0.00001)  # minimalny learning rate

early_stop = EarlyStopping(min_delta = 1, 
                           patience = 20,
                           restore_best_weights = True)

epochs = 1000#trial.suggest_int('epochs',5,30)
learning_rate = 1.#trial.suggest_float('learning_rate',10e-5,10e-3,log = True)
def objective(trial):
    gc.collect()
    
    warstwy = trial.suggest_int('warstwy',1,5)
    filtry = trial.suggest_int('filtry',2,5)
    strides = trial.suggest_int('strides',1,3)
    kernel_size = trial.suggest_int('kernel_size',2,3)
    increase_filters = trial.suggest_int('increase_filters',2,4)
    aktywacja = trial.suggest_categorical('aktywacja',['sigmoid','tanh','elu','relu','linear'])
    aktywacja_out = trial.suggest_categorical('aktywacja_out',['sigmoid','tanh','elu','relu','linear'])
    pooling = trial.suggest_categorical('pooling',['max','average'])
    lags = trial.suggest_int('lags',2,30)
    if pooling =='max':
        pool_fun = GlobalMaxPool2D()
    else:
        pool_fun = GlobalAveragePooling2D()
    lags = trial.suggest_int('lags',5,30)
    #rec_h_std = pd.DataFrame(StandardScaler().fit_transform(rec_h),columns = rec_h.columns,index=rec_h.index)
    rec_l_x,rec_l_y = recursive_array(rec_l,lags)
    rec_l_x = tf.expand_dims(rec_l_x,axis=2)
    train_l_x, test_l_x,train_l_y, test_l_y = rec_l_x[:int(.8*len(rec_l_x))],rec_l_x[int(.8*len(rec_l_x)):],rec_l_y[:int(.8*len(rec_l_x))],rec_l_y[int(.8*len(rec_l_x)):]
    x = Input((lags,1,3))
    out = siec_konwolucyjna(x,warstwy,filtry,strides,kernel_size,aktywacja,pool_fun,pooling_type = 'global',out_act = aktywacja_out,increase_filters =increase_filters)
    model = Model(x,out)
    model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
    # Callback do zapisywania najlepszych modeli
    print(train_l_x.shape)
    history = model.fit(train_l_x,train_l_y,validation_split=.2, 
                        epochs=epochs, batch_size=16, 
                        verbose=True, callbacks=[TerminateOnNaN(),early_stop,reduce_lr])
    gc.collect()
    pred = model.predict(test_l_x)
    if np.any(np.isnan(pred)):
        return history.history['val_mae']
    else:
        return mae(test_l_y,pred)

storage_name = 'sqlite:///cdpt_conv.db'
study = optuna.create_study(study_name = 'cdpr_low',direction='minimize', storage=storage_name, load_if_exists=True)
study.optimize(objective,n_trials=1000)
print(study.best_params)

