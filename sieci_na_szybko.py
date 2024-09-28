
from tensorflow.keras.layers import LSTM,Dense,Conv2D,Flatten,Activation,BatchNormalization
# Funkcje do stworzenia różnych rodzajów sieci "na szybko" ograniczone do kilku najważniejszych parametrów 
def siec_rekurencyjna(x,warstwy,neurony,aktywacja_LSTM,aktywacja_rekurencji,out_act = 'linear'):

    if type(neurony)==int:
        neurony = [neurony for _ in range(warstwy)]
    if type(aktywacja_LSTM)==str:
        aktywacja_LSTM = [aktywacja_LSTM for _ in range(warstwy)]
    if type(aktywacja_rekurencji)==str:
        aktywacja_rekurencji = [aktywacja_rekurencji for i in range(warstwy)]
    
    for i in range(warstwy):
        if i == warstwy-1:
            x = LSTM(units=neurony[i],activation=aktywacja_LSTM[i],recurrent_activation=aktywacja_rekurencji[i],return_sequences=False)(x)
        else:
            print(aktywacja_LSTM[i])
            x = LSTM(units=neurony[i],activation=aktywacja_LSTM[i],recurrent_activation=aktywacja_rekurencji[i],return_sequences=True)(x)
    out = Dense(units = 1,activation = out_act)(x)
    
    return out

def siec_konwolucyjna(x,warstwy,filtry,strides,kernel_size,aktywacja,pooling,pooling_type = 'global',out_act = 'linear'):

    if type(filtry)==int:
        filtry = [filtry for _ in range(warstwy)]
    if type(strides)==int:
        strides = [strides for _ in range(warstwy)]
    if type(aktywacja)!=list:
        aktywacja = [aktywacja for _ in range(warstwy)]
    if pooling_type!='global' and type(pooling)!=list:
        pooling = [pooling for _ in range(pooling)]
    for i in range(warstwy):
        
        x = Conv2D(filters=filtry[i],kernel_size = kernel_size,strides = strides[i],padding = 'same')(x)
        x = aktywacja(x)
        if pooling_type!='global':
            x = pooling[i](x)
    if pooling_type=='global':
        x = pooling(x)
    x = Flatten()(x)
    out = Dense(units = 1,activation = out_act)(x)
    
    return out

def MLP(x,warstwy,neurony,aktywacja,batch_normal):

    if type(neurony)==int:
        neurony = [neurony for _ in range(warstwy)]
    if type(aktywacja)!=list:
        aktywacja = [aktywacja for _ in range(warstwy)]

    for i in range(warstwy):
        if batch_normal:
            x = BatchNormalization()(x)
        x = Dense(units=neurony[i])(x)
        x = Activation(aktywacja[i])(x)
    out = Dense(1,activation = 'linear')(x)
    
    return out