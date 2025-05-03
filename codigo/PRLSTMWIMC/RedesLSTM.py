# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf


# %%
df = pd.read_csv('SOLUSTDAtas_tratado.csv')
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
# # Redes neuronales LSTM

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import LSTM

# %%
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, 3])  
    return np.array(X), np.array(y)

# %%
def preparar_datosLSTM(df, numhoras):
    return create_sequences(df[['open', 'high', 'low', 'close']].values, numhoras)

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
def opti_redes_LSTM(epoch_array, batch_array, X_trainLSTM, y_trainLSTM, X_valiLSTM, y_valiLSTM, X_testLSTM, y_testLSTM, numhoras):
    epoch_best = 0
    bacth_best = 0
    best_model = None
    best = float('inf')
    results = []
    for e in epoch_array:
        for b in batch_array:
            best_value_of_the25 = float('inf')
            for i in range(8):
                #with tf.device('/CPU:0'):
                modelLSTM = Sequential()
                modelLSTM.add(LSTM(128, activation='relu', input_shape=(numhoras, 4)))
                modelLSTM.add(Dense(1))
                modelLSTM.compile(optimizer='adam', loss='mape')
                historyLSTM = modelLSTM.fit(X_trainLSTM, y_trainLSTM, epochs=e, batch_size=b, validation_data=(X_valiLSTM, y_valiLSTM), shuffle=False)
                y_pred = modelLSTM.predict(X_testLSTM)
                valor = evalRedLSTM(y_testLSTM, y_pred)
                print("epoch: "+str(e)+", batch_size: "+str(b)+", value: "+str(valor))
                if valor < best_value_of_the25:
                    best_value_of_the25 = valor
                    if valor < best:
                        epoch_best = e
                        bacth_best = b
                        best_model = modelLSTM
                        best = valor
                    if valor < 0.75:
                        cadena_guardado = "ModelosLSTMOptiMoreDataIMC/mi_modelo_LSTMOpti_e"+str(e)+"_b"+str(b)+"_v"+str(round(valor, 3))
                        best_model.save(cadena_guardado+".keras")
            results.append([e, b, best_value_of_the25])
    df_results = pd.DataFrame(results, columns=["epoch", "batch_size", "value"])
    return epoch_best, bacth_best, best, best_model, df_results

# %%
BATCH_ARRAY = [4, 8, 12, 16, 32, 64, 128, 256]

# %%
def opti_rLSTM_h(h_array, epoch_array, batch_array):
    best = 100
    epoch_best = 0
    bacth_best = 0
    h_best = 0
    best_model = None
    for i in h_array:
        Xtrain, ytrain = preparar_datosLSTM(df_train, i)
        Xvali, yvali = preparar_datosLSTM(df_vali, i)
        Xtest, ytest = preparar_datosLSTM(df_test, i)
        valores = opti_redes_LSTM(epoch_array, batch_array, Xtrain, ytrain, Xvali, yvali, Xtest, ytest, i)
        df = valores[4]
        csv_filename = "lstmH" + str(i) + ".csv"
        df.to_csv(csv_filename, index=False)
    return best, epoch_best, bacth_best, h_best, best_model

# %%
data = opti_rLSTM_h([14, 18, 21], [4, 6, 10, 14, 20, 40], BATCH_ARRAY)

