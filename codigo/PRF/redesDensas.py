# %%
import pandas as pd
import numpy as np
import tensorflow as tf


# %%
tf.config.set_visible_devices([], 'GPU')
print("Dispositivos tras deshabilitar GPUs:", tf.config.get_visible_devices())

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
def opti_redes_densas_multi_gpu(epoch_array, batch_array, numhoras, X_train, y_train, X_vali, y_vali, X_test, y_test):
    best = 100
    epoch_best = 0
    bacth_best = 0
    best_model = None
    training_results = []

    for e in epoch_array:
        for b in batch_array:
            best_value_of_the25 = 100
            best_model_of_the25 = None
            with tf.device('/CPU:0'):
                for m in range(15): # Número de veces que se entrena cada modelo
                    with strategy.scope():
                        model = Sequential()
                        model.add(Dense(128, activation='relu', input_shape=(numhoras * 5,)))
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
                
                training_results.append({"epoch": e, 
                                        "batch_size": b, 
                                        "hours": numhoras, 
                                        "value": best_value_of_the25})
                
            with open('pasosdados.txt', 'w') as archivo:
                archivo.write("epoch: "+str(e)+", batch_size:" + str(b))
            if best_value_of_the25 < best:
                best = best_value_of_the25
                epoch_best = e
                bacth_best = b
                best_model = best_model_of_the25
                if best < 0.75:
                    cadena_guardado = f"ModelosDensosOptiMultiGPUIMC/mi_modelo_densoIMC_Opti_e{e}_b{b}_v{round(best, 3)}_nh{numhoras}"
                    best_model.save(cadena_guardado + ".keras")

    results_df = pd.DataFrame(training_results)
    cadena = "desnsasH" + str(numhoras) + ".csv"
    results_df.to_csv(cadena, index=False)
    print("Resultados guardados en 'densas.csv'")
    return epoch_best, bacth_best, best, best_model


# %%
def opti_rd_h(h_array, epoch_array, batch_array):
    best = 100
    epoch_best = 0
    bacth_best = 0
    h_best = 0
    best_model = None
    for i in h_array:
        Xtrain, ytrain = preparar_datos(df_train, i)
        Xvali, yvali = preparar_datos(df_vali, i)
        Xtest, ytest = preparar_datos(df_test, i)
        valores = opti_redes_densas_multi_gpu(epoch_array, batch_array, i, Xtrain, ytrain, Xvali, yvali, Xtest, ytest)
        if valores[2] < best:
            best = valores[2]
            epoch_best = valores[0]
            bacth_best = valores[1]
            h_best = i
            best_model = valores[3]
            cadena_guardado = "ModelosDensosOptiMoreDataIMCBest/mi_modelo_densoIMC_Opti_e"+str(epoch_best)+"_b"+str(bacth_best)+"_h"+str(i)+"_v"+str(round(best, 3))+"_nh"+str(i)
            best_model.save(cadena_guardado+".keras")
        with open('pasosdadoshoras.txt', 'w') as archivo:
            archivo.write("horas: "+str(i)+"\n")
    return best, epoch_best, bacth_best, h_best, best_model

# %%
#data = opti_rd_h([1, 3, 5, 7, 10, 12, 14, 18, 21], [4, 6, 10, 14, 20, 40], [4, 8, 12, 16, 32, 64, 128, 256])
data = opti_rd_h([3, 21], [4, 6, 10, 14, 20, 40], [4, 8, 12, 16, 32, 64, 128, 256])

print(data)
print("Ha terminado")
