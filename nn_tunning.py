import warnings
import gc
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from tensorflow.keras.layers import Input,GlobalMaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import ReduceLROnPlateau,EarlyStopping,Callback
import optuna
from sklearn.metrics import mean_absolute_error as mae


import numpy as np
from sieci_na_szybko import siec_konwolucyjna


warnings.filterwarnings('ignore')

class TerminateOnNaN(Callback):
    
    """
    Callback to terminate training when the loss becomes NaN.

    This callback monitors the `loss` value after each batch. If a NaN value is 
    detected, the training process is stopped immediately.

    Example:
        model.fit(
            x_train, y_train,
            epochs=10,
            callbacks=[TerminateOnNaN()]
        )
    """
    
    def on_batch_end(self, batch, logs=None):
        
        if (logs.get('loss') is not None and np.isnan(logs.get('loss'))):
            
            print(
                "There are NaNs in loss functions values.\
                Training was terminated"
                )
            
            self.model.stop_training = True


def recursive_array(df, lags):
    
    """
    Generate array of autoregressive features

    Parameters
    ----------
    df : pd.DataFrame
        Original timeseries
    lags : _type_
        The number of delays determining how distant the time 
        horizon should be

    Returns
    -------
    tuple
        2-elements tuple of numpy arrays. First array is features array,
        second is target vector
    """
    
    a = df.values
    x = []
    y=[]
    
    for i in range(len(df) - lags):
        
        x.append(a[i : lags +i ])
        y.append(a[lags + i, :2])
        
    return np.array(x), np.array(y)

def download_data(
    start="2022-01-01", end=(date.today() - timedelta(1)).strftime("%Y-%m-%d"), 
    ticker="CDR.WA", train_split=None
    ):
    """
    Downloading data from yahoofinance.com

    Parameters
    ----------
    start : str, optional
        Start date, by default "2022-01-01"
    end : tuple, optional
        End date, 
        by default (date.today() - timedelta(1)).strftime("%Y-%m-%d")
    ticker : str, optional
        Ticker connected with financial asset interesting
        for us, by default "CDR.WA"
    train_split : _type_, optional
        Splitting parameter - defines how part of
        data will be used to training, by default None

    Returns
    -------
    tuple or pd.DataFrame
        High, Low, Open and close price and volume
        defined asset. Can be splitted on training and testing
        data
    """

    data = yf.download(ticker, start, end)
    rec = data[['High', 'Low', 'Open', 'Close', 'Volume']]
    
    if train_split == None:
        
        return rec
    
    else:
        
        train = rec.iloc[:int(train_split*len(rec))]
        test = rec.iloc[int(train_split*len(rec)):]
        
        return train, test
 
    
def build_model(data,params):    
    """
    Function building and fitting model by 
    given hyperparameters

    Parameters
    ----------
    data : pd.DataFrame
        Data which be used to fitting model
    params : dict
        Dictionary of hyperparamteres which will be
        tuned. Target for differents architectures available
        in tensorflow framework, but now only for convolutional
        neural networks 

    Returns
    -------
    tuple
        Tuple of tensorflow.callbacks.History object 
        and tensorflow.models.Model object
    """
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
                                factor=0.1,  
                                patience=1,  
                                min_lr=0.00001) 

    early_stop = EarlyStopping(min_delta = 1, 
                            patience = 20,
                            restore_best_weights = True)

    # Both values epochs and learning rate are set, because Callbacs're fitting them during learning.
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
        
    rec_x,rec_y = recursive_array(data, lags)
    rec_x = np.expand_dims(rec_x, axis=3)
    x = Input((lags, 5, 1))
    
    out = siec_konwolucyjna(
        x, warstwy, filtry, strides, kernel_size, aktywacja, pool_fun,
        pooling_type='global', out_act=aktywacja_out,
        increase_filters=increase_filters, out_units=2
        )
    
    model = Model(x,out)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), 
        loss=MeanSquaredError(), metrics=[RootMeanSquaredError()]
        )
    
    history = model.fit(
        rec_x, rec_y, validation_split=.2, epochs=epochs, batch_size=16, 
        verbose=True, callbacks=[TerminateOnNaN(), early_stop, reduce_lr]
        )
    
    gc.collect()
    
    return history, model

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")

    train, test = download_data(train_split=.8)
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=1, min_lr=0.00001
        ) 

    early_stop = EarlyStopping(
        min_delta = 1, patience = 20, restore_best_weights=True
        )

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
        
        aktywacja = trial.suggest_categorical(
            'aktywacja',['sigmoid','tanh','elu','relu','linear']
            )
        
        aktywacja_out = trial.suggest_categorical(
            'aktywacja_out',['sigmoid','tanh','elu','relu','linear']
            )
        
        pooling = trial.suggest_categorical('pooling',['max','average'])
        lags = trial.suggest_int('lags',2,30)
        
        params = dict(
            warstwy=warstwy, filtry=filtry, strides=strides, 
            kernel_size=kernel_size, increase_filters=increase_filters,
            aktywacja=aktywacja, aktywacja_out=aktywacja_out,
            pooling=pooling, lags=lags
            )

        test_x,test_y = recursive_array(test,lags)
        test_x = np.expand_dims(test_x,axis=3)
        
        gc.collect()
        history, model = build_model(train, params)
        pred = model.predict(test_x)
        
        if np.any(np.isnan(pred)):
            
            return history.history['val_mae']
        
        else:
            
            return mae(test_y,pred)



    storage_name = 'sqlite:///cdpr_conv.db'
    
    study = optuna.create_study(
        study_name='cdpr_both', direction='minimize', 
        storage=storage_name, load_if_exists=True
        )
    
    study.optimize(objective,n_trials=10)
    
    print(study.best_params)

