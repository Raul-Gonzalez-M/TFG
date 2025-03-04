# %%
import pandas as pd
import numpy as np
import random

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
        for j in range(random.randint(2, numGenes)):   # Se genera un número aleatorio de genes entre 2 y numGenes
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
            return self.__create_priv(4*self.n - 1, numGenes)
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
            return i / j
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
        numberC = (1 + len(candidato)) / 2
        #for i in range(0,20):
        if numberC < self.maxSize and random.randint(0,50) == 0:
            op_cat = random.randint(0, len(self.operations) - 1)
            candidato.append(self.operations[op_cat][random.randint(0, len(self.operations[op_cat]) - 1)])
            candidato.append(random.randrange(0, 4*self.n))
            numberC += 2
        elif numberC > self.minSize and random.randint(0,50) == 0:
            indice = random.randrange(0, len(candidato) - 1)
            del candidato[indice]
            del candidato[indice]
            numberC -= 2
        return candidato
    
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
            nombre = "pandas/dataframe" + str(ronda) + ".csv"
            df.to_csv(nombre, index=False) 
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



    def runcopy(self, numGenes, X, y, baremo: float):
        resultado = []
        num_veces = 0
        best = 1000000
        candidato_best = []
        self.genes = self.create(numGenes)
        for i in range (0, len(self.genes)):
            value = self.fitness2(X, y, self.genes[i])
            if(value < best):
                best = value
                candidato_best = self.genes[i]
        while(best > baremo and num_veces < 1000000):
            for j in range (0, len(self.genes)):
                self.genes[j] = self.mutate2(self.genes[j])
            dicc_aux = {}
            genes_aux = self.genes.copy()
            values_list = []
            for r in range (0, len(self.genes)):
                value = self.fitness2(X, y, self.genes[r])
                values_list.append(value)
                dicc_aux[value] = r
                if(value < best):
                    best = value
                    candidato_best = self.genes[r]
            values_array = np.array(values_list)
            values_array.sort()
            if(num_veces > 200):
                aux = self.genes[dicc_aux[values_array[0]]]
                for ind in range(1,20):
                    indice =  dicc_aux[values_array[values_array.size - ind]]
                    del self.genes[indice]
                for rs in range(1,20):
                    self.genes.append(aux)
            num_veces += 1
            print(f"Vez num:{num_veces}, valor{best}")
            genBest = self.display(candidato_best)
            resultado.append({'iteracion' : num_veces, 'valor' : best, 'gen' : genBest})
            with open('genesIteracion.txt', 'a') as archivo:
                archivo.write(f"Vez num:{num_veces}, valor{best}, gen: {genBest} \n")
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
    maxSize=80,                                     # Tamaño máximo del cromosoma
    minSize=3,                                      # Tamaño mínimo del cromosoma
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
    X.append(aux)
    y.append(df_train.iloc[i + NUMHORAS].close)

# %%
pos = objeto_regresion.runcopy(200, X, y, 0.5)

# %%
pos

# %%
with open('ecuaciones.txt', 'a') as archivo:
    archivo.write(str(pos[0]) + " : " + str(pos[1]) + "_nh" + str(NUMHORAS) + "\n")


