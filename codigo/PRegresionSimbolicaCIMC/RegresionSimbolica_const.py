# %%
import pandas as pd
import numpy as np
import  re
import random

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
class position:
    def __init__(self, value):
        self.value = value
    
    def getValue(self, valores : list[float]):
        return valores[self.value]
    
    def display(self):
        return ("p"+ str(self.value))

# %%
class constant:
    def __init__(self, value):
        self.value = value
    
    def getValue(self, valores : list[float]):
        return self.value
    
    def display(self):
        return ("c"+ str(self.value))

# %%
def aniadirNum(n:int):
        if random.randint(0,2) == 0:
            return constant(random.uniform(0.0, 5.0))
        else:
            return position(random.randrange(0, n))

# %%
def _constant(n):
    return constant(n)

def _position(n):
    return position(n)

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
    
    def create_gen(self, r : int):
        aux = []
        aux.append(aniadirNum(r))
        for i in range(random.randint(self.minSize, self.maxSize - 1)):
            numLO = random.randint(0, len(self.operations) - 1)
            aux.append(self.operations[numLO][random.randint(0, len(self.operations[numLO]) - 1)])
            aux.append(aniadirNum(r))
        return aux
    
    
    def __create_priv(self, r : int, numGenes: int):
        genes = []
        for j in range(0, numGenes):
            aux = self.create_gen(r)
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
            return i / j
        elif op == '^':
            return i ** j
        
    def __evaluate(self, valores: list[float], candidato: list):
        candidato_aux = []
        for i in range(0, len(candidato) - 1, 2):
            candidato_aux.append(candidato[i].getValue(valores))
            candidato_aux.append(candidato[i+1])
        candidato_aux.append(candidato[len(candidato) - 1].getValue(valores))
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
    
    
    def fitness2(self, X, y, candidato: list):
        valores_generados = []
        for elem in X:
            valores_generados.append(self.__evaluate(elem, candidato))
        return self.funcionOptimizacion(valores_generados, y)
               
   
    def display(self,  candidato: list):
        cadena = ""
        for i in range(0, len(candidato)):
            if i % 2 == 1:
                cadena = cadena + candidato[i] + ' '
            else:
                cadena = cadena + candidato[i].display() + ' '
        return cadena
        
    
    def mutate2(self, candidato: list):
        numberC = (1 + len(candidato)) / 2
        n = random.randint(0,1)
        if numberC < self.maxSize and n == 0:
            op_cat = random.randint(0, len(self.operations) - 1)
            candidato.append(self.operations[op_cat][random.randint(0, len(self.operations[op_cat]) - 1)])
            candidato.append(aniadirNum(5*self.n))
            numberC += 2
        elif numberC > self.minSize and n == 1:
            indice = random.randrange(0, len(candidato) - 1)
            del candidato[indice]
            del candidato[indice]
            numberC -= 2
        return candidato
        
        
    def run2(self, numGenes, X, y, baremo: float):
        num_veces = 0
        best = 1000000
        candidato_best = []
        self.genes = self.create(numGenes)
        for i in range (0, len(self.genes)):
            value = self.fitness2(X, y, self.genes[i])
            if(value < best):
                best = value
                candidato_best = self.genes[i]
        while(best > baremo and num_veces < 8001):
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
            indice =  dicc_aux[values_array[values_array.size - 1]]
            del self.genes[indice]
            aux = self.create_gen(4*self.n - 1)
            self.genes.append(aux)
            num_veces += 1
            print(best)
        print("El mejor gen tiene un rendimiento de " + str(best))
        self.display(candidato_best)
        return (candidato_best, best)
    
    def cargar(self, linea):
        gen = []
        tokens = re.findall(r'c\d+(?:\.\d+)?|p\d+|[+\-*/()]', linea)
        for tok in tokens:
            if tok.startswith('c'):
                gen.append(_constant(float(tok[1:])))
            elif tok.startswith('p'):
                gen.append(_position(int(tok[1:])))
            else:
                gen.append(tok)
        return gen
    
    def runcopy(self, numGenes, X, y, baremo: float, cargar:bool):
        resultado = []
        num_veces = 2375
        best = 1000000
        candidato_best = []
        self.genes = []
        if cargar:
            with open('estado.txt', 'r') as archivo:
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
        while(best > baremo and num_veces < 100000):
            best_iteracion = float('inf')
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
            if num_veces > 200 and num_veces % 5 == 0:
                indexed_fitness = list(enumerate(values_list))
                indexed_fitness.sort(key=lambda x: x[1])  # Ordenar por fitness

                mejor_indice = indexed_fitness[0][0]
                aux = self.genes[mejor_indice]

                # Obtener los índices de los 2 peores
                peores_indices = [idx for idx, _ in indexed_fitness[-2:]]
                peores_indices.sort(reverse=True)

                for indice in peores_indices:
                    del self.genes[indice]

                for _ in range(2):
                    self.genes.append(aux)
            num_veces += 1
            print(f"Vez num:{num_veces}, valorit{best_iteracion}, valor{best}")
            genBest = self.display(candi_it)
            resultado.append({'iteracion' : num_veces, 'valor' : best_iteracion, 'gen' : genBest})
            with open('genesIteracion.txt', 'a') as archivo:
                archivo.write(f"Vez num:{num_veces}, valor{best_iteracion}, gen: {genBest} \n")
            if num_veces % 25 == 0:
                df_resultados = pd.DataFrame(resultado)
                cadena = "Dataframes/resultados_regresionSimbolicaCIMC_it" + str(num_veces) + ".csv"
                df_resultados.to_csv(cadena, index=False)
                with open('estado.txt', 'w') as archivo_estado:
                    archivo_estado.write(str(num_veces))
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
    maxSize=15,                                     # Tamaño máximo del cromosoma
    minSize=3,                                      # Tamaño mínimo del cromosoma
    n=NUMHORAS                                             # Cantidad de horas anteriores a considerar
)

# %%

candidato = [constant(3), '+', position(3), '*', constant(5)]
for i in range(0, 300):
    objeto_regresion.mutate2(candidato)
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
with open('ecuaciones_const.txt', 'a') as archivo:
    archivo.write(str(pos[0]) + " : " + str(pos[1]) + "_nh" + str(NUMHORAS) + "\n")


