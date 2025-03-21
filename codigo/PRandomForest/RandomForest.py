# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sea
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import joblib


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
# # Declaramos y Entrenamos el modelo

# %% [markdown]
# Función que prepara los datos

# %%
def preparar_datosRandomForest(df, numhoras):
    X = []
    y= []
    for i in range(0, df.shape[0] - numhoras):
        auxy = df.iloc[i + numhoras]
        y.append(auxy.close)
        aux1 = []
        for e in range(0, numhoras):
            aux = df.iloc[i + e]
            for r in range(1, aux.size):
                aux1.append(aux[r])
        X.append(aux1)
    return (X, y)

# %% [markdown]
# Preparamos los datos en el formato requerido por el algoritmo para su entrenamiento

# %%
data = preparar_datosRandomForest(df_train, numhorasconst)
X1 = data[0]
y1 = data[1]

# %% [markdown]
# Preparamos los datos usados para testear el algortimo en el formato requerido

# %%
data = preparar_datosRandomForest(df_valitest, numhorasconst)
Xvt = data[0]
yvt = data[1]

# %% [markdown]
# Desarrollamos un algoritmo para entrenar distintas instacias con distintos parámetros y compararlas

# %%
def evalRandomForest(Testrpr, predictT):
    suma = 0
    n = len(Testrpr)
    for i in range(0,n):
        suma = abs(predictT[i] - Testrpr[i])/Testrpr[i] +  suma
    error_medio = suma/n
    emp = error_medio*100 # error medio en porcentaje
    return emp

# %%
def train_randomForestdepth(d_array, Xtr, ytr, Xvtaux, yvtaux, numhoras):
    resultados = []
    posbest = 0
    best = 100
    for i in d_array:
        regr1 = RandomForestRegressor(max_depth=i, random_state=0)
        regr1.fit(Xtr, ytr)
        predictT = regr1.predict(Xvtaux)
        valor = evalRandomForest(yvtaux, predictT)
        resultados.append({'numhoras' : numhoras,'max_depth': i, 'valor': valor})
        if valor < best:
            best = valor
            posbest = i
            if valor < 0.8:
                cadena = "Modelos/random_forest_model_h" + str(numhoras) + "_d" + str(i) + ".pkl"
                joblib.dump(regr1, cadena)
    return(posbest, best, resultados)
        

# %%
def train_randomForest(h_array, d_array):
    resultados = []
    posbest = 0
    best = 100
    for i in h_array:
        Xtrain, ytrain = preparar_datosRandomForest(df_train, i)
        Xvtaux, yvtaux = preparar_datosRandomForest(df_valitest, i)
        valores = train_randomForestdepth(d_array, Xtrain, ytrain, Xvtaux, yvtaux, i)
        valor = valores[1]
        resultados.extend(valores[2])
        print(str(i)+" "+str(valores[0])+" "+str(valor))
        with open('OptimizaciónRandomForestIMC.txt', 'a') as archivo:
            archivo.write("Número de horas: "+str(i)+" Profundidad: "+str(valores[0])+" Valor de emp obtenido: "+str(valor) + "\n")
        if valor < best:
            best = valor
            posbest = valores[0]
    df_resultados = pd.DataFrame(resultados)
    cadena = "Dataframes/resultados_randomForest.csv"
    df_resultados.to_csv(cadena, index=False)
    return(posbest, best)

# %%
tuple = train_randomForest([1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,75,100],[1,2,3,4,5,6,7,8,9,10,14,18,20,30,40,50,75,100,150,200,300,400])

# %%
print(tuple[0])
print(tuple[1])


