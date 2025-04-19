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
def create_df_n(df, n):
    df_aux = df.copy()
    for i in range(1, n):
        df_aux['open_before' + str(i)] = df_aux['open'].shift(+i)
        df_aux['high_before' + str(i)] = df_aux['high'].shift(+i)
        df_aux['low_before' + str(i)] = df_aux['low'].shift(+i)
        df_aux['close_before' + str(i)] = df_aux['close'].shift(+i)
        df_aux['value_before' + str(i)] = df_aux['value'].shift(+i)
    df_aux['close_next'] = df_aux['close'].shift(-1)
    df_aux = df_aux.dropna()
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
    tamanio_aux = df_aux.shape[0]
    return df_aux.copy().iloc[0:int(tamanio_aux*0.7)]

# %%
def createdfvali(df_aux):
    tamanio_aux = df_aux.shape[0]
    return df_aux.copy().iloc[int(tamanio_aux*0.7 + 1):int(tamanio_aux*0.9)]

# %%
def createdftest(df_aux):
    tamanio_aux = df_aux.shape[0]
    return df_aux.copy().iloc[int(tamanio_aux*0.9 + 1):tamanio_aux]

# %%
def preparar_datosXGBoost(df_aux):
    X = df_aux.drop(['date', 'close_next'], axis=1)
    #X = df_aux[['open', 'high', 'low', 'close', 'open_before1', 'high_before1', 'low_before1', 'close_before1']] 
    y = df_aux['close_next']
    return (xgb.DMatrix(data=X, label=y), y)

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
                best_depth = i  # Guardo la profundidad del mejor modelo hasta el moemnto
                if best < 0.75: # Si el rendimiento es menor que 0.75 guardo el modelo
                    cadena = "Modelos/modelo_xgb_v.json" + str(valor)+ "_d" + str(i) + "_eta" + str(param.get("eta")) + ".json"
                    bstaux.save_model(cadena) # Guardo el modelo
    return (best_depth, best, resultados)

# %%
def trainGlobalXGB(d_array, h_array):
    best = 100
    best_depth = 0
    for i in h_array:
        df_aux = create_df_n(df, i)
        dtrain_aux = createdftrain(df_aux)
        dvali_aux = createdfvali(df_aux)
        dtest_aux = createdftest(df_aux)
        dtrain_prep = preparar_datosXGBoost(dtrain_aux)
        dvali_prep = preparar_datosXGBoost(dvali_aux)
        dtest_prep = preparar_datosXGBoost(dtest_aux)
        values = train_XGB_depth(d_array, dtrain_prep[0], dvali_prep[0], dtest_prep[0], dtest_prep[1].values)
        print(str(i)+" "+str(values[0])+" "+str(values))
        df_resultados = pd.DataFrame(values[2])
        cadena = "Dataframes/resultados_xgboost_h" + str(i) + ".csv"
        df_resultados.to_csv(cadena, index=False)
        with open('OptimizaciónXGBoostIMC.txt', 'a') as archivo:
            archivo.write("Numero de horas: "+str(i)+" Profundidad: "+str(values[0])+" Valor de emp obtenido: "+str(values[1]) + "\n")
        if(values[1] < best):
            best = values[1]
            best_depth = values[0]
    return best, best_depth

# %%
data = trainGlobalXGB([1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,75,100],[1,2,3,4,5,6,7,8,9,10,14,18,20,30,40,50,75,100])


