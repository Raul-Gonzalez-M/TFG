# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf


# %%
df = pd.read_csv('SolAtasIMC_tratado.csv')
df.head()

# %%
print(df.info())

# %% [markdown]
# # Preprocesado de Datos

# %%
tamanio = df.shape[0]

# %%
df_train = df.copy().loc[0:int(tamanio*0.7)]
df_train

# %%
df_vali = df.copy().loc[int(tamanio*0.7 + 1):int(tamanio*0.9)]
df_vali

# %%
df_test = df.copy().loc[int(tamanio*0.9 + 1):tamanio]
df_test

# %%
df_valitest = pd.concat([df_vali, df_test], axis=0)

# %% [markdown]
# Numero de horas que se utilizan en la predicción

# %%
numhorasconst = 4

# %% [markdown]
# # Redes neuronales Densas

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# %%
print("¿GPU detectada?:", tf.config.list_physical_devices('GPU'))

# %%
print("Versión de TensorFlow:", tf.__version__)

# %%
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, 3])  
    return np.array(X), np.array(y)

# %%
def preparar_datos(df, numhoras):
    data = df[['open', 'high', 'low', 'close', 'value']].values
    X, y = create_sequences(data, numhoras)
    X_aux = []
    for i in X:
        aux = []
        for r in range(0, numhoras):
            for elem in i[r]:
                aux.append(elem)
        X_aux.append(aux)       
    X_aux = np.array(X_aux) 
    return X_aux, y

# %%
def evalRedDensa(ytest, y_pred):
    y_pred = y_pred.flatten()
    suma = 0
    n = len(y_pred)
    for i in range(0,n):
        suma = abs(y_pred[i] - ytest[i])/ytest[i] +  suma
    error_medio = suma/n
    emp = error_medio*100 # error medio en porcentaje
    return emp

# %%
strategy = tf.distribute.MirroredStrategy()
print(f"Número de GPUs detectadas: {strategy.num_replicas_in_sync}")

# %%
def opti_redes_densas_multi_gpu(epoch_ini, epoch_fin, batch_array, numhoras, X_train, y_train, X_vali, y_vali, X_test, y_test):
    best = 100
    epoch_best = 0
    bacth_best = 0
    best_model = None

    for e in range(epoch_ini, epoch_fin + 1):
        for b in batch_array:
            best_value_of_the25 = 100
            best_model_of_the25 = None

            for i in range(0, 25):  # Número de veces que se entrena cada modelo
                with strategy.scope():
                    model = Sequential()
                    model.add(Dense(64, activation='relu', input_shape=(numhoras * 5,)))
                    model.add(Dense(64, activation='relu'))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mape')

                history = model.fit(X_train, y_train, epochs=e, batch_size=b, validation_data=(X_vali, y_vali), shuffle=False)
                y_pred = model.predict(X_test)
                valor = evalRedDensa(y_test, y_pred)

                if valor < best_value_of_the25:
                    best_value_of_the25 = valor
                    best_model_of_the25 = model

            print(f"Epoch: {e}, Batch size: {b}, Value: {best_value_of_the25}")
            with open('pasosdados.txt', 'w') as archivo:
                archivo.write("epoch: "+str(e)+", batch_size:" + str(b))

            if best_value_of_the25 < best:
                best = best_value_of_the25
                epoch_best = e
                bacth_best = b

                cadena_guardado = f"ModelosDensosOptiMultiGPU/mi_modelo_denso_Opti_e{e}_b{b}_v{round(best_value_of_the25, 3)}_nh{numhoras}"
                best_model_of_the25.save(cadena_guardado + ".h5")
                best_model_of_the25.save(cadena_guardado + ".keras")
                best_model = best_model_of_the25

    return epoch_best, bacth_best, best, best_model


# %%
def opti_rd_h(inih, finh, epoch_ini, epoch_fin, batch_array):
    best = 100
    epoch_best = 0
    bacth_best = 0
    h_best = 0
    best_model = None
    for i in range(inih, finh+1):
        Xtrain, ytrain = preparar_datos(df_train, i)
        Xvali, yvali = preparar_datos(df_vali, i)
        Xtest, ytest = preparar_datos(df_test, i)
        valores = opti_redes_densas_multi_gpu(epoch_ini, epoch_fin, batch_array, i, Xtrain, ytrain, Xvali, yvali, Xtest, ytest)
        if valores[2] < best:
            best = valores[2]
            epoch_best = valores[0]
            bacth_best = valores[1]
            h_best = i
            best_model = valores[3]
            cadena_guardado = "ModelosDensosOptiMoreDataIMCBest/mi_modelo_denso_Opti_e"+str(epoch_best)+"_b"+str(bacth_best)+"_h"+str(i)+"_v"+str(round(best, 3)+"_nh"+str(i))
            best_model.save(cadena_guardado+".h5")
            best_model.save(cadena_guardado+".keras")
        with open('pasosdadoshoras.txt', 'w') as archivo:
            archivo.write("horas: "+str(i)+"\n")
    return best, epoch_best, bacth_best, h_best, best_model

# %%
data = opti_rd_h(7, 16, 3, 15, [4, 6, 8, 12, 16, 24, 32, 46, 64, 96, 128, 256])

# %% [markdown]
# # Redes neuronales LSTM

# %%
from tensorflow.keras.layers import LSTM
import random

# %%
df[['open', 'high', 'low', 'close', 'value']] = df[['open', 'high', 'low', 'close', 'value']].astype('float32')


# %%
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, 3])  
    return np.array(X), np.array(y)

# %%
def preparar_datosLSTM(df, numhoras):
    return create_sequences(df[['open', 'high', 'low', 'close', 'value']].values, numhoras)

# %%
def evalRedLSTM(ytest, y_pred):
    y_pred = y_pred.flatten()
    suma = 0
    n = len(y_pred)
    for i in range(0,n):
        suma = abs(y_pred[i] - ytest[i])/ytest[i] +  suma
    error_medio = suma/n
    emp = error_medio*100 # error medio en porcentaje
    return emp

# %%
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# %%
def opti_redes_LSTM_multi_gpu(epoch_ini, epoch_fin, batch_array, X_trainLSTM, y_trainLSTM, X_valiLSTM, y_valiLSTM, X_testLSTM, y_testLSTM, numhoras):
    best = 100
    epoch_best = 0
    bacth_best = 0
    best_model = None
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        for e in range(epoch_ini, epoch_fin + 1):
            for b in batch_array:
        
                modelLSTM = Sequential()
                modelLSTM.add(LSTM(64, activation='relu', input_shape=(numhoras, 5), dtype='float32'))
                modelLSTM.add(Dense(1, dtype='float32'))  # Forzar la salida a ser float32
                modelLSTM.compile(optimizer='adam', loss='mape')

                modelLSTM.summary()

                historyLSTM = modelLSTM.fit(X_trainLSTM, y_trainLSTM, epochs=e, batch_size=b, validation_data=(X_valiLSTM, y_valiLSTM), shuffle=False)
                y_pred = modelLSTM.predict(X_testLSTM)
                valor = evalRedLSTM(y_testLSTM, y_pred)

                print(f"Epoch: {e}, Batch size: {b}, Value: {valor}")

                if valor < best:
                    best = valor
                    epoch_best = e
                    bacth_best = b
                    best_model = modelLSTM

                    cadena_guardado = f"ModelosLSTMOptiMultiGPU/mi_modelo_LSTMOpti_e{e}_b{b}_v{round(valor, 3)}"
                    best_model.save(cadena_guardado + ".h5")
                    best_model.save(cadena_guardado + ".keras")

    return epoch_best, bacth_best, best, best_model

# Ajustar parámetros para entrenamiento distribuido


# %%
BATCH_ARRAY = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128, 192, 256]
BATCH_ARRAY = np.array(BATCH_ARRAY, dtype=int)

# %%
def opti_rLSTM_h(inih, finh, epoch_ini, epoch_fin, batch_array):
    best = 100
    epoch_best = 0
    bacth_best = 0
    h_best = 0
    best_model = None
    for i in range(inih, finh+1):
        Xtrain, ytrain = preparar_datosLSTM(df_train, i)
        Xvali, yvali = preparar_datosLSTM(df_vali, i)
        Xtest, ytest = preparar_datosLSTM(df_test, i)
        Xtrain = Xtrain.astype('float32')
        ytrain = ytrain.astype('float32')
        Xvali = Xvali.astype('float32')
        yvali = yvali.astype('float32')
        Xtest = Xtest.astype('float32')
        ytest = ytest.astype('float32')
        valores = opti_redes_LSTM_multi_gpu(epoch_ini, epoch_fin, batch_array, Xtrain, ytrain, Xvali, yvali, Xtest, ytest, i)
        if valores[2] < best:
            best = valores[2]
            epoch_best = valores[0]
            bacth_best = valores[1]
            h_best = i
            best_model = valores[3]
            cadena_guardado = "ModelosLSTMOptiMoreDataIMCBest/mi_modelo_LSTM_Opti_e"+str(epoch_best)+"_b"+str(bacth_best)+"_h"+str(i)+"_v"+str(round(best, 3))
            best_model.save(cadena_guardado+".h5")
            best_model.save(cadena_guardado+".keras")
    return best, epoch_best, bacth_best, h_best, best_model

# %%
data = opti_rLSTM_h(7, 16, 3, 20, BATCH_ARRAY)


