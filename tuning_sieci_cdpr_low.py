import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.layers import Input,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import ReduceLROnPlateau,EarlyStopping
import numpy as np
from sieci_na_szybko import siec_rekurencyjna
import gc

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

early_stop = EarlyStopping(min_delta = 0.001, patience = 20)
aktywacja_lstm = 'tanh'#trial.suggest_categorical('aktywacja_lstm',['sigmoid','tanh'])#,
aktywacja_rek = 'sigmoid'#trial.suggest_categorical('aktywacja_rek',['sigmoid','tanh','elu','relu','linear'])
epochs = 1000#trial.suggest_int('epochs',5,30)
learning_rate = 1.#trial.suggest_float('learning_rate',10e-5,10e-3,log = True)
def objective(trial):
    gc.collect()
    l_warstw = trial.suggest_int('l_warstw',1,10)
    l_neuronow = trial.suggest_int('l_neuronow',1,3)
    
    aktywacja_out = trial.suggest_categorical('aktywacja_out',['sigmoid','tanh','elu','relu','linear'])
    
    lags = trial.suggest_int('lags',5,30)
    #rec_h_std = pd.DataFrame(StandardScaler().fit_transform(rec_h),columns = rec_h.columns,index=rec_h.index)
    rec_l_x,rec_l_y = recursive_array(rec_l,lags)
    train_l_x, test_l_x,train_l_y, test_l_y = rec_l_x[:int(.8*len(rec_l_x))],rec_l_x[int(.8*len(rec_l_x)):],rec_l_y[:int(.8*len(rec_l_x))],rec_l_y[int(.8*len(rec_l_x)):]
    x = Input((lags,3))
    x = BatchNormalization()(x)
    out = siec_rekurencyjna(x,l_warstw,l_neuronow,aktywacja_lstm,aktywacja_rek,aktywacja_out)
    model = Model(x,out)
    model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss=MeanAbsoluteError(), metrics=['mae'])
    checkpoint_dir = 'checkpoints2'
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Callback do zapisywania najlepszych modeli
    filepath = f"checkpoints2/model_trial_{trial.number}.weights.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_weights_only=True,  # Można zmienić na False, aby zapisać cały model
            save_best_only=True,     # Zapisuje tylko, gdy model poprawia wyniki
            monitor='val_loss',      # Metryka, którą monitorujemy
            mode='min',              # Chcemy minimalizować stratę
            verbose=1
        )
    history = model.fit(train_l_x,train_l_y,validation_split=.2, 
                        epochs=epochs, batch_size=254, 
                        verbose=True, callbacks=[TerminateOnNaN(),checkpoint_callback,early_stop,reduce_lr])
    gc.collect()
    pred = model.predict(test_l_x)
    if np.any(np.isnan(pred)):
        return history.history['val_mae']
    else:
        return mae(test_l_y,pred)

storage_name = 'sqlite:///optuna_study_new_env.db'
study = optuna.create_study(study_name = 'cdpr_low',direction='minimize', storage=storage_name, load_if_exists=True)
study.optimize(objective,n_trials=1000)
print(study.best_params)
import pickle
sorted_trials = sorted(study.trials, key=lambda t: t.value)
for i in range(5):
    with open(f'best_params_low{i+1}.pickle','wb') as file:
        pickle.dump([sorted_trials[i].values, sorted_trials[i].params],file)
