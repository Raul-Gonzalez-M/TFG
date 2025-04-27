# %%
import pandas as pd
import numpy as np
import random
import re



# %%
df = pd.read_csv('SOLUSTDAtas_tratado.csv')
print(df.info())

# %% [markdown]
# # Preprocesado de Datos

# %%
tamanio = df.shape[0]

# %%
NUMHORAS = 7

# %%
df_train = df.copy().loc[0:int(tamanio*0.7)]
df_vali = df.copy().loc[int(tamanio*0.7 + 1):int(tamanio*0.9)]
df_test = df.copy().loc[int(tamanio*0.9 + 1):tamanio]

# %% [markdown]
# # Desarrollo del modelo

# %%
class RegresionSimbolica:
    def __init__(self, funcionOptimizacion: callable, operations: list[list], maxSize: int, minSize: int, n : int): 
        # Asigno a funcionOptimizacion la función que se debe utilizar para calcular el rendimiento de los genes
        self.funcionOptimizacion = funcionOptimizacion  
        self.operations = operations  # Asigno a operations la lista de listas de operaciones
        self.maxSize = maxSize  # Asigno a maxSize el tamañao máximo que puede tener un gen
        self.minSize = minSize  # Asigno a minSize el tamaño mínimo que puede tener un gen
        self.genes: list[list] = [] # Asigno a genes una lista vacía
        self.n = n  # Asigno a n la cantidad de horas anteriores que se tienen en cuenta
        pass
    
    
    def __create_priv(self, r : int, numGenes: int):
        genes = []
        for j in range (numGenes):   
            aux = []
            aux.append(random.randint(0, r)) # Añado al gen un número aleatorio entre 0 y la posición máxima
            # Genero un tamaño aleatorio para el gen comprendido entre el tamaño máximo y el tamaño mínimo
            for i in range(random.randint(self.minSize, self.maxSize - 1)): 
                # Selecciono aleatoriamente una categoría de operación
                numLO = random.randint(0, len(self.operations) - 1) 
                # Añado al gen una operación elegida aleatoriamente de entre las de la categoría 
                aux.append(self.operations[numLO][random.randint(0, len(self.operations[numLO]) - 1)])  
                # Añado al gen un número aleatorio entre 0 y la posición máxima
                aux.append(random.randint(0, r))
            genes.append(aux)   # Añado el gen generado a la lista de genes
        return genes
            
    
    
    def create(self, numGenes: int):
        if numGenes > 0:    # Compruebo que el número de genes sea al menos 1
            return self.__create_priv(4*self.n - 1, numGenes)
        else:
            # Levanto una excepción porque el número de genes es menor que 1
            raise Exception("El número de genes no puede ser menor que 1")  
        
    def __aplica_operacion(self, op, i, j):
        if op == '+':   # Compruebo si la operación es la suma
            return i + j    # Sumo i más j
        elif op == '-': # Compruebo si la operación es la resta
            return i - j    # Resto i menos j
        elif op == '*': # Compruebo si la operación es la multiplicación
            return i * j    # Multiplico i por j
        elif op == '/': # Compruebo si la operación es la división
            return i / j if j != 0 else 1e6 # Si j es distinto de 0 divido i entre j
        elif op == '^': # Compruebo si la operación es la potencia
            return i ** j   # Elevo i a j
        
    def __evaluate(self, valores: list[float], candidato: list):
        candidato_aux = []
        # Recorro el gen y sustituyo las posiciones por sus valores correspondientes creando una lista auxiliar
        for i in range(0, len(candidato) - 1, 2):  
            candidato_aux.append(valores[candidato[i]]) # Añado el valor correspondiente a esa posición
            candidato_aux.append(candidato[i+1])    # Añado la operación
        candidato_aux.append(valores[candidato[len(candidato) - 1]]) # Añado el valor correspondiente a esa posición
        # Recorro las categorías de operaciones ejecutando las operaciones de cada categoría
        for categoria_operaciones in self.operations:   
            j_offset = 0    # offset del índice
            # Recorro las operaciones del gen
            for j in range(1, len(candidato_aux), 2):
                indice = j + j_offset
                op = candidato_aux[indice]  # Asigno la operación a op
                # Compruebo si la operación esta en al categoría de operaciones de esta iteración
                if(op in categoria_operaciones): 
                    # Realizo la operación
                    op_result = self.__aplica_operacion( op, candidato_aux[indice-1], candidato_aux[indice+1])
                    candidato_aux[indice-1] = op_result # Asigno el resultado de la operación al lugar del operador izquierdo
                    del candidato_aux[indice+1] # Elimino la operación del gen
                    del candidato_aux[indice]   # Elimino el operador derecho del gen
                    j_offset = j_offset - 2 # Reduzco en 2 el offset
        return candidato_aux[0]   # Devuelvo el valor obtenido
    
    
    def fitness2(self, X, y, candidato: list):
        valores_generados = []
        for elem in X: # Recorro X obteniendo la predicción del modelo para cada entrada de X
            valores_generados.append(self.__evaluate(elem, candidato))
        # Ejecuto al función de optimización en al predicción e y
        return self.funcionOptimizacion(valores_generados, y)   
               
                
        
            
        
    def display(self,  candidato: list):
        # Decuelvo el gen como string
        return  ' '.join(map(str, [i for i in candidato]))
        
    def mutate(self, candidato: list):
        numberC = (1 + len(candidato)) / 2
        if numberC < self.maxSize and random.randint(0,100) == 0:
            op_cat = random.randint(0, len(self.operations) - 1)
            candidato.append(self.operations[op_cat][random.randint(0, len(self.operations[op_cat]) - 1)])
            candidato.append(random.randrange(0, 4*self.n))
            return candidato
        elif numberC > self.minSize and random.randint(0,50) == 0:
            indice = random.randrange(0, len(candidato) - 1)
            del candidato[indice]
            del candidato[indice]
            return candidato
        return candidato
    
    def mutate2(self, candidato: list):
        numberC = (1 + len(candidato)) / 2  # Tamaño del gen
        # Si el tamaño del gen no es mayor que el tamaño máximo y el número aleatorio es 0 el gen se muta añadiendo
        if numberC < self.maxSize and random.randint(0,49) == 0:    
            op_cat = random.randint(0, len(self.operations) - 1) # Selecciono aleatoriamente una categoría de operación
            # Añado al gen una operación elegida aleatoriamente de entre las de la categoría 
            candidato.append(self.operations[op_cat][random.randint(0, len(self.operations[op_cat]) - 1)])
            # Añado al gen un número aleatorio entre 0 y la posición máxima
            candidato.append(random.randrange(0, 4*self.n))
            # Si no se ha añadido, el tamaño del gen menos 2 no es menor que el tamaño mínimo y el número aleatorio es 0 el gen se muta quitando
        elif numberC - 2 > self.minSize and random.randint(0,49) == 0:
            # Selecciono aleatoriamente un índice del gen
            indice = random.randrange(0, len(candidato) - 1)
            #Elimino el elemento que está en al posición del índice
            del candidato[indice]
            #Elimino el elemento que está en al posición del índice
            del candidato[indice]
        return candidato    # Devuelvo el gen resultante

    def cargar(self, linea):
        gen = []
        gen = re.findall(r'\d+|[+\-*/]', linea)
        gen = [int(tok) if tok.isdigit() else tok for tok in gen]
        return gen



    def runcopy(self, numGenes, X, y, baremo: float, cargar: bool):
        resultado = []
        num_veces = 0
        best = 1000000
        candidato_best = []
        self.genes = []
        if cargar:
            with open('estado_generado2.txt', 'r') as archivo:
                primera_linea = archivo.readline()
                num_veces = int(primera_linea.strip())
                for linea in archivo:
                    linea = linea.strip()
                    self.genes.append(self.cargar(linea))
                numGenes = len(self.genes)
        else:
            self.genes = self.create(numGenes)
        for i in range (0, len(self.genes)):
            value = self.fitness2(X, y, self.genes[i])
            if(value < best):
                best = value
                candidato_best = self.genes[i]
        while(best > baremo and num_veces < 1000000):
            best_iteracion = 1000000
            candi_it = []
            for j in range (0, len(self.genes)):
                self.genes[j] = self.mutate2(self.genes[j])
            values_list = []
            for r in range (0, len(self.genes)):
                value = self.fitness2(X, y, self.genes[r])
                values_list.append(value)
                if(value < best_iteracion):
                    best_iteracion = value
                    candi_it = self.genes[r]
                    if value < best:
                        best = value
                        candidato_best = self.genes[r]
            if num_veces > 200 and num_veces % 100 == 0:
                indexed_fitness = list(enumerate(values_list))
                indexed_fitness.sort(key=lambda x: x[1])  # Ordeno los genes según su rendimiento
                mejor_indice = indexed_fitness[0][0]
                aux = self.genes[mejor_indice]
                peores_indices = [idx for idx, _ in indexed_fitness[-5:]]
                peores_indices.sort(reverse=True)
                for indice in peores_indices:
                    del self.genes[indice]
                for _ in range(5):
                    self.genes.append(aux)
            num_veces += 1
            print(f"Vez num:{num_veces}, valorit{best_iteracion}, valor{best}")
            genBest = self.display(candi_it)
            resultado.append({'iteracion' : num_veces, 'valor' : best_iteracion, 'gen' : genBest})
            with open('genesIteracion.txt', 'a') as archivo:
                archivo.write(f"Vez num:{num_veces}, valor{best_iteracion}, gen: {genBest} \n")
            if num_veces % 100 == 0:
                df_resultados = pd.DataFrame(resultado)
                cadena = "Dataframes/resultados_regresionSimbolicaC_it" + str(num_veces) + ".csv"
                df_resultados.to_csv(cadena, index=False)
                with open('estado.txt', 'w') as archivo_estado:
                    for gen in self.genes:
                        cadena  = str(self.display(gen)) + "\n"
                        archivo_estado.write(cadena)
        print("El mejor gen tiene un rendimiento de " + str(best))
        self.display(candidato_best)
        df_resultados = pd.DataFrame(resultado)
        cadena = "Dataframes/resultados_regresionSimbolica.csv"
        df_resultados.to_csv(cadena, index=False)
        return (candidato_best, best)


# %%
def funcion_optimizacion_mape(valores_generados, y):
    suma = 0
    n = len(valores_generados)
    for i in range(0,n):
        suma = abs(valores_generados[i] - y[i])/y[i] +  suma
    error_medio = suma/n
    return error_medio*100 

# %%

operations = [
    ['*', '/'],   # Primera lista de operaciones prioritarias
    ['+', '-']    # Segunda lista de operaciones con menor prioridad
]

# Declarar el objeto
objeto_regresion = RegresionSimbolica(
    funcionOptimizacion=funcion_optimizacion_mape,  # Pasas tu función de optimización
    operations=operations,                          # Pasas la lista de operaciones
    maxSize=50,                                     # Tamaño máximo del cromosoma
    minSize=5,                                      # Tamaño mínimo del cromosoma
    n=NUMHORAS                                            # Cantidad de horas anteriores a considerar
)

# %%
print(objeto_regresion.display([1, '+', 32, '*', 5, '-', 6, '+', 9, '/', 2]))
objeto_regresion.fitness([[1, 3,1,4,5,2,5,8,5]], [10.2], [1, '+', 3, '*', 5])
candidato = [1, '+', 3, '*', 5]
for i in range(0, 300):
    objeto_regresion.mutate(candidato)
objeto_regresion.display(candidato)    



# %% [markdown]
# Preparar X e y

# %%
X = []
y= []
for i in range(0, df_train.shape[0] - NUMHORAS):
    aux = []
    for r in range(0, NUMHORAS):
        aux.append(df_train.iloc[i + r].open)
        aux.append(df_train.iloc[i + r].high)
        aux.append(df_train.iloc[i + r].low)
        aux.append(df_train.iloc[i + r].close)
    X.append(aux)
    y.append(df_train.iloc[i + NUMHORAS].close)

# %%
pos = objeto_regresion.runcopy(200, X, y, 0.5, True)

# %%
pos

# %%
with open('ecuaciones.txt', 'a') as archivo:
    archivo.write(str(pos[0]) + " : " + str(pos[1]) + "_nh" + str(NUMHORAS) + "\n")


