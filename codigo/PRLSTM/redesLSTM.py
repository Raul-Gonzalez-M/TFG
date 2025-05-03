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
# # Redes neuronales LSTM

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import LSTM

# %%
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])     # Añado a X las listaa de listas de precios
        y.append(data[i+n_steps, 3])    # Añado a y el precio de cierre de la siguiente hora  
    return np.array(X), np.array(y)     # Los transformo en arrays de numpy

# %%
def preparar_datosLSTM(df, numhoras):
    # Llamo a la función create sequences con los valores del df en un array de numpy y el número de horas
    return create_sequences(df[['open', 'high', 'low', 'close', 'value']].to_numpy(), numhoras)

# %%
def evalRedLSTM(ytest, y_pred):
    y_pred = y_pred.flatten()   # Transforma una lista de listas en una lista de valores
    suma = 0
    n = len(y_pred) # Obtengo el tamaño de y
    for i in range(0,n):    # Suma el error relativo de todas las predicciones
        suma = abs(y_pred[i] - ytest[i])/ytest[i] +  suma
    error_medio = suma/n    # Divido la suma entre el número de predicciones para calcular la media
    emp = error_medio*100   # Multiplico por 100 para obtener el error porcentual medio
    return emp

# %%
def opti_redes_LSTM(epoch_array, batch_array, X_trainLSTM, y_trainLSTM, X_valiLSTM, y_valiLSTM, X_testLSTM, y_testLSTM, numhoras):
    best_model = None
    best = float('inf') # Asigno a best un valor infinito de tipo float
    results = []
    for e in epoch_array:   # Genero modelos con todos los epoch pasados como parámetro en el array
        for b in batch_array:   # Genero modelos con todos los batch_size pasados como parámetro en el array
            best_value_of_the15 = float('inf')  # Asigno a best_value_of_the15 un valor infinito de tipo float
            for i in range(8): # Genero 15 modelos con cada de par epoch, bacth_size 
                with tf.device('/CPU:0'):   # Fuerzo que el código debajo de esta línea se ejecute usando la CPU
                    modelLSTM = Sequential()    # Declaro el modelo como secuencial
                    # Creo la capa de entrada con 128 neuronas, definiendo la forma de la entrada y y cuya función de activacion es Rectified Linear Unit
                    modelLSTM.add(LSTM(128, activation='relu', input_shape=(numhoras, 5)))
                    # Creo la capa de salida
                    modelLSTM.add(Dense(1))
                    # Compilo la red neuronal usando adam como optimizador y mape, Mean Absolute Percentage Error, como función que debe minimizar
                    modelLSTM.compile(optimizer='adam', loss='mape')
                    # Entreno el modelo
                    historyLSTM = modelLSTM.fit(X_trainLSTM, y_trainLSTM, epochs=e, batch_size=b, validation_data=(X_valiLSTM, y_valiLSTM), shuffle=False)
                    # Realizo una predicción usando los datos de test
                    y_pred = modelLSTM.predict(X_testLSTM)
                    valor = evalRedLSTM(y_testLSTM, y_pred) # Evalúo el rendimiento del modelo
                    print("epoch: "+str(e)+", batch_size: "+str(b)+", value: "+str(valor))  # Imprimo la epoch, el batch_size y el rendimiento obtenido
                    if valor < best_value_of_the15: # Si el rendimiento que obtengo es el mejor de esta iteración hasta ahora lo sustituyo
                        best_value_of_the15 = valor
                        if valor < best:    # Si el rendimiento que he obtenido es el mejor hasta ahora lo sustituyo y guardo el modelo en la variable best_model
                            best_model = modelLSTM
                            best = valor
                        if valor < 0.75:    # Si el rendimiento es mejor que 0,75 guardo el modelo
                            # Creo el nombre del modelo y lo guardo en la variable cadena_guardado
                            cadena_guardado = "ModelosLSTMOptiMoreDataIMC/mi_modelo_LSTMOpti_e"+str(e)+"_b"+str(b)+"_v"+str(round(valor, 3))
                            best_model.save(cadena_guardado+".keras")   # Guardo el modelo
            results.append([e, b, best_value_of_the15]) # Guardo la información de la epoch, el batch_size y el mejor rendimiento obtenido para estos
    df_results = pd.DataFrame(results, columns=["epoch", "batch_size", "value"])    # Transformo los datos guardados en un dataframe
    return df_results   # Devuelvo el dataframe

# %%
BATCH_ARRAY = [4, 8, 12, 16, 32, 64, 128, 256]

# %%
def opti_rLSTM_h(h_array, epoch_array, batch_array):
    for i in h_array:
        Xtrain, ytrain = preparar_datosLSTM(df_train, i)    # Preparo los datos en el formato requerido para el entrenamiento
        Xvali, yvali = preparar_datosLSTM(df_vali, i)   # Preparo los datos en el formato requerido para la validación
        Xtest, ytest = preparar_datosLSTM(df_test, i)   # Preparo los datos en el formato requerido para el test
        # Guardo en df el dataframe que decuelve
        df = opti_redes_LSTM(epoch_array, batch_array, Xtrain, ytrain, Xvali, yvali, Xtest, ytest, i) 
        csv_filename = "lstmH" + str(i) + ".csv"    # Guardo en csv_filename el nombre con el que voy a guardar el dataframe de df
        df.to_csv(csv_filename, index=False)    # Guardo el dataframe
    return "He terminado"

# %%
# Llamo a la función de entrenamiento principal, la primera es la lista con las horas, al segunda es la lista con las epoch y la tercera es la lista de los batch_size
data = opti_rLSTM_h([14, 18, 21], [4, 6, 10, 14, 20, 40], BATCH_ARRAY)


