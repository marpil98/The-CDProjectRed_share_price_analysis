import pandas as pd
import yfinance as yf
import warnings
from sieci_na_szybko import siec_rekurencyjna,siec_konwolucyjna,MLP
from tensorflow.keras.layers import Input,BatchNormalization

from sieci_na_szybko import siec_rekurencyjna,siec_konwolucyjna,MLP
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae

import tensorflow as tf
import numpy as np

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

def objective(trial):
    l_warstw = trial.suggest_int('l_warstw',1,10)
    l_neuronow = trial.suggest_int('l_neuronow',1,10)
    aktywacja_lstm = trial.suggest_categorical('aktywacja_lstm',['sigmoid','tanh'])#,
    aktywacja_rek = trial.suggest_categorical('aktywacja_rek',['sigmoid','tanh','elu','relu','linear'])
    aktywacja_out = trial.suggest_categorical('aktywacja_out',['sigmoid','tanh','elu','relu','linear'])
    epochs = trial.suggest_int('epochs',5,30)
    learning_rate = trial.suggest_float('learning_rate',10e-5,10e-3,log = True)
    lags = trial.suggest_int('lags',10,30)
    #rec_h_std = pd.DataFrame(StandardScaler().fit_transform(rec_h),columns = rec_h.columns,index=rec_h.index)
    
    rec_h_x,rec_h_y = recursive_array(rec_h,lags)
    train_h_x, test_h_x,train_h_y, test_h_y = rec_h_x[:int(.8*len(rec_h_x))],rec_h_x[int(.8*len(rec_h_x)):],rec_h_y[:int(.8*len(rec_h_x))],rec_h_y[int(.8*len(rec_h_x)):]
    x = Input((lags,3))
    x = BatchNormalization()(x)
    out = siec_rekurencyjna(x,l_warstw,l_neuronow,aktywacja_lstm,aktywacja_rek,aktywacja_out)
    model = Model(x,out)
    model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss=MeanAbsoluteError(), metrics=['mae'])
    history = model.fit(train_h_x,train_h_y,validation_split=.2, epochs=epochs, batch_size=32, verbose=True, callbacks=[TerminateOnNaN()])
    pred = model.predict(test_h_x)
    if np.any(np.isnan(pred)):
        return history.history['val_mae']
    else:
        return mae(test_h_y,pred)


study = optuna.create_study(direction='minimize')
study.optimize(objective,n_trials=5000)
print(study.best_params)
import pickle
sorted_trials = sorted(study.trials, key=lambda t: t.value)
for i in range(5):
    with open(f'best_params{i+1}.pickle','wb') as file:
        pickle.dump([sorted_trials[i].values, sorted_trials[i].params],file)
