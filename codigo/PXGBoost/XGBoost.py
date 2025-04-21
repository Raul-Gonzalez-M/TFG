# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import xgboost as xgb


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
# # Predicción Utilizando XGBoost

# %%
def create_df_n(df, n): # Creo un nuevo dataframe con un formato adecuado para el XGBoost
    df_aux = df.copy()
    for i in range(1, n):   # Añado en el dataframe que he creado los valores de hace n - 1 horas como columnas 
        df_aux['open_before' + str(i)] = df_aux['open'].shift(+i) # Por ejemplo el valor de apertura de hace 2 horas se añade como open_before2
        df_aux['high_before' + str(i)] = df_aux['high'].shift(+i)
        df_aux['low_before' + str(i)] = df_aux['low'].shift(+i)
        df_aux['close_before' + str(i)] = df_aux['close'].shift(+i)
        df_aux['value_before' + str(i)] = df_aux['value'].shift(+i)
    df_aux['close_next'] = df_aux['close'].shift(-1) # Añado en el dataframe el valor de cierre de la siguiente hora 
    df_aux = df_aux.dropna()    # Elimino todas las filas que no tengan todos los valores
    return df_aux

# %% [markdown]
# Ponemos los datos en el formato correcto

# %%
df_xgb = create_df_n(df, 1)


# %%
tamanio_xgb = df_xgb.shape[0]
tamanio_xgb

# %%
def createdftrain(df_aux):
    tamanio_aux = df_aux.shape[0]   # Obtengo el tamaño del dataframe
    return df_aux.copy().iloc[0:int(tamanio_aux*0.7)]   # Selecciono el 70% inicial para ser el dataframe para training

# %%
def createdfvali(df_aux):
    tamanio_aux = df_aux.shape[0]   # Obtengo el tamaño del dataframe
    return df_aux.copy().iloc[int(tamanio_aux*0.7 + 1):int(tamanio_aux*0.9)]    # Selecciono los datos entre el 70% y el 90% para ser el dataframe para training

# %%
def createdftest(df_aux):
    tamanio_aux = df_aux.shape[0]   # Obtengo el tamaño del dataframe
    return df_aux.copy().iloc[int(tamanio_aux*0.9 + 1):tamanio_aux] # Selecciono el 10% final para ser el dataframe para training

# %%
def preparar_datosXGBoost(df_aux):
    X = df_aux.drop(['date', 'close_next'], axis=1) # Elimino la columna date y la columna close_next
    y = df_aux['close_next']    # Asigno a y los valores de la columna close_next
    return (xgb.DMatrix(data=X, label=y), y)    # Convierto X en una matriz

# %%
df_train_xgb = createdftrain(df_xgb)
df_train_xgb

# %%
df_vali_xgb = createdfvali(df_xgb)
df_vali_xgb

# %%
df_test_xgb = createdftest(df_xgb)
df_test_xgb

# %%
df_valitest_xgb = pd.concat([df_vali_xgb, df_test_xgb], ignore_index=True)

# %%
XvtXGB = df_valitest_xgb[['open', 'high', 'low', 'close', 'value']] 
yvtXGB = df_valitest_xgb['close_next']

# %%
dtrainvali = xgb.DMatrix(data=XvtXGB, label=yvtXGB)

# %% [markdown]
# Definimos las características para el entrenamiento

# %% [markdown]
# Comprobación de la tasa de error en los datos de test

# %%
def evalXGB(Test_xgb, predict_xgb_test):
    suma = 0
    n = len(Test_xgb)
    for i in range(0,n):    # Suma el error relativo de todas las predicciones.
        suma = abs(predict_xgb_test[i] - Test_xgb[i])/Test_xgb[i] +  suma
    error_medio = suma/n    # Divido la suma entre el número de predicciones para calcular la media
    emp = error_medio*100   # Multiplico por 100 para obtener el error porcentual medio
    return emp

# %%
def train_XGB_depth(d_array, dtrainf, dvalif, dtestf, ytest):
    resultados = []
    best = float('inf') # Variable en la que guardo el mejor resultado
    best_depth = 0  # Variable en la que guardo la profundidad que genera el mejor resultado
    for i in d_array:   # Genero un modelo para todas las profundidades
        etaAux = [0.3 , 0.1, 0.01]  # Diferentes parámetro eta que vamos a utilizar
        for e in etaAux:    # Genero un modelo para todas las eta
            param = {'max_depth': i, 'eta': e, 'objective': 'reg:squarederror'} # Asigno los parámetros para el modelo
            evals = [(dtrainf, 'train'), (dvalif, 'validacion')]  # Asigno los datos para entrenar y validar el modelo
            esr = int(1//e) # Calculo el parámetro early stoping rounds en función del valor de eta que este usando
            if(esr < 10): # Si el parámetro early stoping rounds es menor que 10 lo cambio a 10
                esr = 10
            # Entreno el modelo
            bstaux = xgb.train(param, dtrainf, num_boost_round=int(100//e), evals=evals, early_stopping_rounds=esr, verbose_eval=100)
            predict_xgb_test = bstaux.predict(dtestf)   # Utilizo el modelo que acabo de entrenar para predecir usando los datos para test
            valor = evalXGB(ytest, predict_xgb_test)    # Evalúo el rendimiento del modelo usando la predicción y los datos reales
            resultados.append({'max_depth': i, 'eta': param['eta'], 'valor': valor})  # Guaro los resultados del rendimiento de este modelo
            if(valor < best):   # Si el rendimiento del modelo es menor que el mejor rendimiento hasta ahora lo asigno como el mejor
                best = valor
                best_depth = i  # Guardo la profundidad del mejor modelo hasta el momento
                if best < 0.75: # Si el rendimiento es menor que 0.75 guardo el modelo
                    cadena = "Modelos/modelo_xgb_v.json" + str(valor)+ "_d" + str(i) + "_eta" + str(param.get("eta")) + ".json"
                    bstaux.save_model(cadena) # Guardo el modelo
    return (best_depth, best, resultados)

# %%
def trainGlobalXGB(d_array, h_array):
    best = float('inf') # Variable en la que guardo el mejor resultado
    best_depth = 0  # Variable en la que guardo la profundidad que genera el mejor resultado
    for i in h_array:
        df_aux = create_df_n(df, i) # Creo el dataframe acorde al número de horas de esta iteración, i
        dtrain_aux = createdftrain(df_aux)  # Creo el dataframe que voy a usar para el entrenamiento
        dvali_aux = createdfvali(df_aux)    # Creo el dataframe que voy a usar para la validación
        dtest_aux = createdftest(df_aux)    # Creo el dataframe que voy a usar para el test
        dtrain_prep = preparar_datosXGBoost(dtrain_aux) # Función con la que preparo los datos que voy a usar para el entrenamiento
        dvali_prep = preparar_datosXGBoost(dvali_aux) # Función con la que preparo los datos que voy a usar para la validación
        dtest_prep = preparar_datosXGBoost(dtest_aux)   # Función con la que preparo los datos que voy a usar para el testeo
        # Entreno los modelos usando los datos preparados con la cantidad de horas correcta
        values = train_XGB_depth(d_array, dtrain_prep[0], dvali_prep[0], dtest_prep[0], dtest_prep[1].values)   
        print(str(i)+" "+str(values[0])+" "+str(values[1]))    # Imprimo por pantalla el número de horas, la mejor profundidad de la iteración y el mejor rendimiento de la iteración
        df_resultados = pd.DataFrame(values[2]) # Convierto los resultados del modelo en un dataframe
        cadena = "Dataframes/resultados_xgboost_h" + str(i) + ".csv"
        df_resultados.to_csv(cadena, index=False)   # Guardo el dataframe
        with open('OptimizaciónXGBoostIMC.txt', 'a') as archivo:
            archivo.write("Numero de horas: "+str(i)+" Profundidad: "+str(values[0])+" Valor de emp obtenido: "+str(values[1]) + "\n")
        if(values[1] < best): # Si el rendimiento obtenido durante el entrenamiento es menor que el mejor rendimiento hasta ahora lo asigno como el mejor
            best = values[1]
            best_depth = values[0]  # Guardo la profundidad del mejor modelo hasta el momento
    return best, best_depth

# %%
# Ejecuto la función principal trainGlobalXGB, la primera lista es la lista que contiene las profundidades y la segunda lista es la lista que contiene las horas 
data = trainGlobalXGB([1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,75,100],[1,2,3,4,5,6,7,8,9,10,14,18,20,30,40,50,75,100])


