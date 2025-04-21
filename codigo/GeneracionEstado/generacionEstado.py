# %%
import pandas as pd
import numpy as np



# %%


# Carga del archivo CSV
df = pd.read_csv("resultados_regresionSimbolicaC_it5600.csv")  # Reemplaza con el nombre correcto

# Eliminar filas duplicadas por la columna 'valor'
df_sin_duplicados = df.drop_duplicates(subset='valor')

# Ordenar por la columna 'valor' de menor a mayor
df_ordenado = df_sin_duplicados.sort_values(by='valor', ascending=True)

# Reiniciar índice (opcional, solo para limpieza)
df_ordenado = df_ordenado.reset_index(drop=True)

print(df_ordenado)


# %%
# Suponiendo que la columna que contiene los genes se llama 'gen'
top_200_genes = df_ordenado['gen'].head(200).tolist()

# Mostrar los primeros 10 por si querés verificar
print(top_200_genes[:10])


# %%
with open("estado_generado2.txt", "w") as f:
    for gen in top_200_genes:
        f.write(f"{gen}\n")



