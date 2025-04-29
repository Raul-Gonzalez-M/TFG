# %%
import pandas as pd
import numpy as np
import random
import re

# %%
df = pd.read_csv('SolAtasIMC_tratado.csv')
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
    def __init__(self, funcionOptimizacion: callable, operations: list[list], maxSize: int, minSize: int, n : int): # n es la cantidad de horas anteriores que se tienen en cuenta
        self.funcionOptimizacion = funcionOptimizacion
        self.operations = operations  # La primera lista contendrá las operaciones más prioritarias y así progresivamente
        self.maxSize = maxSize
        self.minSize = minSize
        self.genes: list[list] = []
        self.n = n
        pass
    
    
    def __create_priv(self, r : int, numGenes: int):
        genes = []
        for j in range(numGenes):   # Se genera un número aleatorio de genes entre 2 y numGenes
            aux = []
            aux.append(random.randint(0, r))
            for i in range(random.randint(self.minSize, self.maxSize - 1)):
                numLO = random.randint(0, len(self.operations) - 1)
                aux.append(self.operations[numLO][random.randint(0, len(self.operations[numLO]) - 1)])
                aux.append(random.randint(0, r))
            genes.append(aux)
        return genes
            
    
    
    def create(self, numGenes: int):
        if numGenes > 0:
            return self.__create_priv(5*self.n - 1, numGenes)
        else:
            raise Exception("El número de genes no puede ser menor que 1")
        
    def __aplica_operacion(self, op, i, j):
        if op == '+':
            return i + j
        elif op == '-':
            return i - j
        elif op == '*':
            return i * j
        elif op == '/':
            return i / j if j != 0 else 1e6
        elif op == '^':
            return i ** j
        
    def __evaluate(self, valores: list[float], candidato: list):
        candidato_aux = []
        for i in range(0, len(candidato) - 1, 2):
            candidato_aux.append(valores[candidato[i]])
            candidato_aux.append(candidato[i+1])
        candidato_aux.append(valores[candidato[len(candidato) - 1]])
        for categoria_operaciones in self.operations:
            j_offset = 0
            for j in range(1, len(candidato_aux), 2):
                indice = j + j_offset
                op = candidato_aux[indice]
                if(op in categoria_operaciones):
                    op_result = self.__aplica_operacion( op, candidato_aux[indice-1], candidato_aux[indice+1])
                    candidato_aux[indice-1] = op_result
                    del candidato_aux[indice+1]
                    del candidato_aux[indice]
                    j_offset = j_offset - 2
        return candidato_aux[0]   
    
    
    def fitness(self, X, y, candidato: list):
        valores_generados = []
        for elem in X:
            valores_generados.append(self.__evaluate(elem, candidato))
        return self.funcionOptimizacion(valores_generados, y)
    
    def fitness2(self, X, y, candidato: list):
        valores_generados = []
        for elem in X:
            valores_generados.append(self.__evaluate(elem, candidato))
        return self.funcionOptimizacion(valores_generados, y)
               
                
        
            
        
    def display(self,  candidato: list):
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
        n = random.randint(0,1) # Genero un 0 o un 1
        # Si el tamaño del gen no es mayor que el tamaño máximo y n es 0 el gen se muta añadiendo
        if numberC < self.maxSize and n == 0:
            op_cat = random.randint(0, len(self.operations) - 1)    # Selecciono aleatoriamente una categoría de operación
            # Añado al gen una operación elegida aleatoriamente de entre las de la categoría 
            candidato.append(self.operations[op_cat][random.randint(0, len(self.operations[op_cat]) - 1)])
            # Añado al gen un número aleatorio entre 0 y la posición máxima
            candidato.append(random.randrange(0, 4*self.n))
        # Si no se ha añadido, el tamaño del gen menos 2 no es menor que el tamaño mínimo y n es 1 el gen se muta quitando 
        if numberC > self.minSize and n == 1:
            # Selecciono aleatoriamente un índice del gen
            indice = random.randrange(0, len(candidato) - 1)
            # Elimino el elemento que está en la posición del índice
            del candidato[indice]
            # Elimino el elemento que está en la posición del índice
            del candidato[indice]
        return candidato    # Devuelvo el gen resultante
    
    def run(self, numGenes, X, y, baremo: float):
        best = 1000000
        post_best = 0
        self.genes = self.create(numGenes)
        for i in range (0, len(self.genes)):
            value = self.fitness(X, y, self.genes[i])
            if(value < best):
                best = value
                post_best = i
        while(best > baremo):
            for j in range (0, len(self.genes)):
                self.genes[j] = self.mutate2(self.genes[j])
            for i in range (0, len(self.genes)):
                value = self.fitness(X, y, self.genes[i])
                if(value < best):
                    best = value
                    post_best = i
        print("El mejor gen tiene un rendimiento de " + str(best))
        self.display(self.genes[post_best])
        return (post_best, best)
        
        
    def run2(self, numGenes, X, y, baremo: float):
        best = 1000000
        candidato_best = []
        self.genes = self.create(numGenes)
        for i in range (0, len(self.genes)):
            value = self.fitness(X, y, self.genes[i])
            if(value < best):
                best = value
                candidato_best = self.genes[i]
        ronda = 0
        while(best > baremo):
            ronda +=1
            for j in range (0, len(self.genes)):
                self.genes[j] = self.mutate2(self.genes[j])
            dicc_aux = {}
            genes_aux = self.genes.copy()
            values_list = []
            for r in range (0, len(self.genes)):
                value = self.fitness(X, y, self.genes[r])
                values_list.append(value)
                dicc_aux[value] = r
                if(value < best):
                    best = value
                    candidato_best = self.genes[r]
            values_array = np.array(values_list)
            values_array.sort()
            #Crear la lista que va a ir en el dataFrame
            genesauxiliares = []
            for value in values_array:
                genesauxiliares.append(self.genes[dicc_aux[value]])
            df = pd.DataFrame({
                'Genes': genesauxiliares,
                'Valor': values_array,
                'NumHoras': NUMHORAS,
                'Ronda': ronda
            })
            nombre = "pandas/dataframe" + str(ronda)
            df.to_csv(nombre + '.csv', index=False) 
            num_del = 0
            if(values_array.size <= (numGenes - 4)/2):
                genescreados = self.create(4)
                for gen in genescreados:
                    self.genes.append(gen)
            else:
                for z in range(int(values_array.size/ 2), values_array.size):
                    if genes_aux[dicc_aux[values_array[z]]] in self.genes:
                        self.genes.remove(genes_aux[dicc_aux[values_array[z]]])
                        num_del += 1
                if num_del > 2:
                    genescreados = self.create(num_del)
                    for gen in genescreados:
                        self.genes.append(gen)
            print("El mejor gen," + str(display(candidato_best)) + ",de la ronda " + str(ronda) + " tiene un rendimiento de " + str(best))
            with open('candidatos.txt', 'a') as archivo:
                archivo.write("El mejor gen," + str(display(candidato_best)) + ",de la ronda " + str(ronda) + " tiene un rendimiento de " + str(best) + "\n")
        print("El mejor gen tiene un rendimiento de " + str(best))
        self.display(candidato_best)
        return (candidato_best, best)
    
    
    def cargar(self, linea):
        gen = []
        gen = re.findall(r'\d+|[+\-*/]', linea)
        gen = [int(tok) if tok.isdigit() else tok for tok in gen]
        return gen



    def runcopy(self, numGenes, X, y, baremo: float, cargar):
        resultado = []  # Lista en la que guardo los resultados de cada iteración
        num_veces = 0   # Variable en la que guardo al cantidad de iteraciones
        best = float('inf') # Variable en la que guardo el mejor resultado
        candidato_best = [] # Variable en la que guardo el mejor gen
        if cargar:  # Si cargar es true la población inicial se obtiene del archivo de texto estado
            with open('estado.txt', 'r') as archivo:    # Abro el archivo en modo lectura
                primera_linea = archivo.readline()  # Leo la primera línea
                num_veces = int(primera_linea.strip())  # Transformo lo leído en un entero y se lo asigno a número de veces
                for linea in archivo:   # Leo cada línea del archivo de una en una
                    linea = linea.strip() # Elimino espacios al principio y al final
                    self.genes.append(self.cargar(linea))   # Transformo la línea leída en un gen y lo añado a la lista de genes
                numGenes = len(self.genes)  # Asigno al número de genes la longitud de la lista de genes
        else:   # Si cargar es false al población inicial se genera aleatoriamente
            self.genes = self.create(numGenes)  # Creo la población inicial y la asigno a la lista de genes
        for i in range (0, len(self.genes)):     # Evalúo el rendimiento de cada gen y obtengo el mejor rendimiento
            value = self.fitness2(X, y, self.genes[i])   # Evalúo el rendimiento del gen
            if(value < best):   # Si es mejor que el mejor hasta ahora lo sustituye
                best = value
                candidato_best = self.genes[i]  # Guardo el gen como el mejor hasta el momento
        while(best > baremo and num_veces < 100000):    # Mientras que el mejor no sea mejor que el baremo y el número de iteraciones sea menor de 100000 se ejecuta el bucle
            best_iteracion = float('inf')   # Variable en la que guardo el mejor resultado de esta iteración
            candi_it = []
            for j in range (0, len(self.genes)):  # Recorro la lista de genes
                self.genes[j] = self.mutate2(self.genes[j]) # Ejecuto mutate2 con el gen correspondiente
            values_list = []
            for r in range (0, len(self.genes)):    # Recorro la lista de genes
                value = self.fitness2(X, y, self.genes[r])   # Evalúo el rendimiento del gen
                values_list.append(value)   # Añado el rendimiento a la lista de rendimientos
                if(value < best_iteracion): # Si es mejor que el mejor de la iteración hasta ahora lo sustituye
                    best_iteracion = value
                    candi_it = self.genes[r]    # Guardo el gen como el mejor de la iteración hasta el momento
                    if value < best:    # Si es mejor que el mejor hasta ahora lo sustituye
                        best = value
                        candidato_best = self.genes[r]  # Guardo el gen como el mejor hasta el momento
            if num_veces > 200 and num_veces % 5 == 0:  # Si el num de iteraciones es mayor de 200 y multiplo de 5 se ejecuta el proceso de eliminación de los peores genes
                indexed_fitness = list(enumerate(values_list))  # Asigno índices a cada valor de la lista de valores y lo guardo como lista de tuplas (índice, valor)
                indexed_fitness.sort(key=lambda x: x[1])  # Ordeno la lista por rendimiento
                mejor_indice = indexed_fitness[0][0]    # Guardo el índice del mejor gen de la lista 
                aux = self.genes[mejor_indice]  # Guardo en aux el mejor gen de la lista
                peores_indices = [idx for idx, _ in indexed_fitness[-2:]]   # Guardo el índice de los dos peores genes de la lista 
                peores_indices.sort(reverse=True)   # Ordeno los índices de mayor a menor
                for indice in peores_indices:   # Recorro la lista de índices
                    del self.genes[indice]  # Elimino el gen que se encuentra en este índice
                for _ in range(2):  # Añado a la lista de genes dos copias de el mejor gen de la lista de genes
                    self.genes.append(aux)
            num_veces += 1  # Sumo 1 al número de iteraciones
            # Imprimo el número de iteraciones, el mejor rendimiento de la iteración y el mejor rendimiento hasta ahora
            print(f"Vez num:{num_veces}, valorit{best_iteracion}, valor{best}")
            genBest = self.display(candi_it)     # Guardo el mejor gen de la iteración como string
            # Guardo el número de iteración, el mejor rendimiento de la iteración y el mejor gen de la iteración como string
            resultado.append({'iteracion' : num_veces, 'valor' : best_iteracion, 'gen' : genBest})
            with open('genesIteracion.txt', 'a') as archivo:    # Abro el archivo en modo append
                # Guardo en el archivo de texto el número de iteración, el mejor rendimiento de la iteración y el mejor gen de la iteración como string
                archivo.write(f"Vez num:{num_veces}, valor{best_iteracion}, gen: {genBest} \n")
            if num_veces % 100 == 0:    # Si el número de iteraciones en múltiplo de 100
                df_resultados = pd.DataFrame(resultado) # Transformo la lista en un dataframe
                # Creo el nombre de guardado del dataframe
                cadena = "Dataframes/resultados_regresionSimbolicaC_it" + str(num_veces) + ".csv"
                df_resultados.to_csv(cadena, index=False)   # Guardo el dataframe como csv
                with open('estado.txt', 'w') as archivo_estado: # Abro el archivo en modo escritura
                    cad = str(num_veces)+ "\n"  # Creo un string con el número de iteraciones y un salto de línea
                    archivo_estado.write(cad)   # Lo escribo en el archivo
                    for gen in self.genes:  # Recorro la lista de genes
                        cadena  = str(self.display(gen)) + "\n" # Creo un string con el gen y un salto de línea
                        archivo_estado.write(cadena)# Lo escribo en el archivo
        print("El mejor gen tiene un rendimiento de " + str(best))  # Imprimo el mejor rendimiento obtenido
        self.display(candidato_best)    # Guardo el mejor gen como string
        df_resultados = pd.DataFrame(resultado) # Guardo los resultados como dataframe
        cadena = "Dataframes/resultados_regresionSimbolica.csv" # Creo el nombre de guardado del dataframe
        df_resultados.to_csv(cadena, index=False)   # Guardo el dataframe como csv
        return (candidato_best, best)   # Devuelvo el mejor gen y su rendimiento

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
print(objeto_regresion.display([1, '+', 3, '*', 5, '-', 6, '+', 9, '/', 2]))
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
        aux.append(df_train.iloc[i + r].value)
    X.append(aux)
    y.append(df_train.iloc[i + NUMHORAS].close)

# %%
pos = objeto_regresion.runcopy(200, X, y, 0.5, False)

# %%
pos

# %%
with open('ecuaciones.txt', 'a') as archivo:
    archivo.write(str(pos[0]) + " : " + str(pos[1]) + "_nh" + str(NUMHORAS) + "\n")


