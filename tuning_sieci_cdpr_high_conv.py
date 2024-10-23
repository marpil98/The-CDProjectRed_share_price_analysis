import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.layers import Input,BatchNormalization,GlobalMaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError,MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import ReduceLROnPlateau,EarlyStopping,Callback
import optuna
from sklearn.metrics import mean_absolute_error as mae


import numpy as np
from sieci_na_szybko import siec_konwolucyjna
import gc

from datetime import date, datetime, timedelta
class TerminateOnNaN(Callback):
    '''
    Class terminating learning, when loss function return NaN
    '''
    def on_batch_end(self, batch, logs=None):
        if logs.get('loss') is not None and np.isnan(logs.get('loss')):
            print("There are NaNs in loss functions values. Training was terminated")
            self.model.stop_training = True


def recursive_array(df, lags):
    '''
    Function creating new date frame, where lagged values are using as features.
    '''
    a = df.values
    x = []
    y=[]
    for i in range(len(df)-lags):
        x.append(a[i:lags+i])
        y.append(a[lags+i,0])
    return np.array(x),np.array(y)

def build_best_model(params):
    start_date = "2022-01-01"
    end_date = (date.today()-timedelta(1)).strftime("%Y-%m-%d")
    ticker = "CDR.WA"

    data = yf.download(ticker,start_date,end_date)
    data.head()
    rec = data[['High','Low','Open','Close']]
    rec_l = rec.drop(columns = 'High')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
                                factor=0.1,  
                                patience=1,  
                                min_lr=0.00001) 

    early_stop = EarlyStopping(min_delta = 1, 
                            patience = 20,
                            restore_best_weights = True)

    # Both values epochs and learning rate are setted, because Callbacs're fittign them during learning.
    epochs = 1000 
    learning_rate = 1.    
    warstwy = params['warstwy']
    filtry = params['filtry']
    strides = params['strides']
    kernel_size = params['kernel_size']
    increase_filters = params['increase_filters']
    aktywacja = params['aktywacja']
    aktywacja_out = params['aktywacja_out']
    pooling = params['pooling']
    lags = params['lags']
    if pooling =='max':
        pool_fun = GlobalMaxPool2D()
    else:
        pool_fun = GlobalAveragePooling2D()
    #rec_h_std = pd.DataFrame(StandardScaler().fit_transform(rec_h),columns = rec_h.columns,index=rec_h.index)
    rec_h_x,rec_h_y = recursive_array(rec_h,lags)
    print(rec_h_x)
    rec_h_x = np.expand_dims(rec_h_x,axis=3)
    x = Input((lags,3,1))
    out = siec_konwolucyjna(x,warstwy,filtry,strides,kernel_size,aktywacja,pool_fun,pooling_type = 'global',out_act = aktywacja_out,increase_filters =increase_filters)
    model = Model(x,out)
    model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
    # Callback to saving best trial
    history = model.fit(rec_h_x,rec_h_y,validation_split=.2, 
                        epochs=epochs, batch_size=16, 
                        verbose=True, callbacks=[TerminateOnNaN(),early_stop,reduce_lr])
    gc.collect()
    return history
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    start_date = "2022-01-01"
    end_date = "2024-09-17"
    ticker = "CDR.WA"

    data = yf.download(ticker,start_date,end_date)
    data.head()
    rec = data[['High','Low','Open','Close']]
    rec_h = rec.drop(columns = 'Low')
    rec_l = rec.drop(columns = 'High')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
                                factor=0.1,  
                                patience=1,  
                                min_lr=0.00001) 

    early_stop = EarlyStopping(min_delta = 1, 
                            patience = 20,
                            restore_best_weights = True)

    # Both values epochs and learning rate are setted, because Callbacs're fittign them during learning.
    epochs = 1000 
    learning_rate = 1.
    def objective(trial):
        gc.collect() # it should clean memory
        
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
        rec_h_x,rec_h_y = recursive_array(rec_h,lags)
        rec_h_x = np.expand_dims(rec_h_x,axis=3)
        train_h_x, test_h_x,train_h_y, test_h_y = rec_h_x[:int(.8*len(rec_h_x))],rec_h_x[int(.8*len(rec_h_x)):],rec_h_y[:int(.8*len(rec_h_y))],rec_h_y[int(.8*len(rec_h_y)):]
        x = Input((lags,3,1))
        out = siec_konwolucyjna(x,warstwy,filtry,strides,kernel_size,aktywacja,pool_fun,pooling_type = 'global',out_act = aktywacja_out,increase_filters =increase_filters)
        model = Model(x,out)
        model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
        # Callback to saving best trial
        print(train_l_x.shape)
        history = model.fit(train_h_x,train_h_y,validation_split=.2, 
                            epochs=epochs, batch_size=16, 
                            verbose=True, callbacks=[TerminateOnNaN(),early_stop,reduce_lr])
        gc.collect()
        pred = model.predict(test_h_x)
        if np.any(np.isnan(pred)):
            return history.history['val_mae']
        else:
            return mae(test_h_y,pred)



    storage_name = 'sqlite:///cdpt_conv.db'
    study = optuna.create_study(study_name = 'cdpr_low_another_dim',direction='minimize', storage=storage_name, load_if_exists=True)
    study.optimize(objective,n_trials=1000)
    print(study.best_params)

