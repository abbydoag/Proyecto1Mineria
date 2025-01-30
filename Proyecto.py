"""
Entrega 1 Analisis explorativo
Abby Donis 22440
Wilson Calderón 22018
"""
import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as plt

#Lee zip
with zipfile.ZipFile('movies.zip', 'r') as z:
    with z.open('movies.csv') as f:
        datos = pd.read_csv(f, encoding='ISO-8859-1')
#delimitacion a 3 decimales
pd.set_option('display.float_format', '{:.3f}'.format)

#Inciso 1
print(datos.describe())


#Inciso 2 (col = columan)
"""
Si la variable:
Es tipo objeto = cualitiativa normal u ordinal
No. finito de categorías = ordinal
-----------------
Es numerica = se revisa si es continua o discreta (revisar si el no. de valores unicos es grande o pequeño)
"""
print('++ Clasificación tipos de variables ++')
print('.......................................')
def variable_type(col):
    if col.dtype == 'object':
       return 'Cualitativa Nominal u Ordinal'
    elif col.dtype in ['int64', 'float64']:
        if len(col.unique()) > 20:
            return 'Cuantitativa Continua'
        else:
            return 'Cuantitativa Discreta'
    else:
        'No se pudo identificar'
for col in datos.columns:
    tipo = variable_type(datos[col])
    print (f'Columna: {col} --- Tipo: {tipo}')
# Inciso 3: Análisis de Normalidad y Tablas de Frecuencia

# Identificar variables cuantitativas
cuantitativas = datos.select_dtypes(include=['int64', 'float64'])

# Análisis de normalidad usando media, desviación estándar y asimetría
print("\n++ Prueba de Normalidad ++")
for col in cuantitativas.columns:
    valores = cuantitativas[col].dropna()
    media = valores.mean()
    std = valores.std()
    
    # Cálculo de la regla empírica (aproximación)
    dentro_1_std = ((valores > media - std) & (valores < media + std)).sum() / len(valores)
    dentro_2_std = ((valores > media - 2*std) & (valores < media + 2*std)).sum() / len(valores)
    dentro_3_std = ((valores > media - 3*std) & (valores < media + 3*std)).sum() / len(valores)
    
    # Calcular asimetría para ver qué tan lejos está de una normal
    asimetria = valores.skew()
    
    print(f"\nVariable: {col}")
    print(f"  - Dentro de 1 desviación estándar: {dentro_1_std:.3f} (Esperado: ~68%)")
    print(f"  - Dentro de 2 desviaciones estándar: {dentro_2_std:.3f} (Esperado: ~95%)")
    print(f"  - Dentro de 3 desviaciones estándar: {dentro_3_std:.3f} (Esperado: ~99.7%)")
    print(f"  - Asimetría: {asimetria:.3f} (Cercano a 0 indica simetría)")

    # Interpretación basada en los resultados
    if abs(dentro_1_std - 0.68) < 0.05 and abs(dentro_2_std - 0.95) < 0.05 and abs(dentro_3_std - 0.997) < 0.05 and abs(asimetria) < 0.5:
        print("  -> La distribución parece normal.")
    else:
        print("  -> La distribución NO parece normal.")

# Identificar variables cualitativas y generar tabla de frecuencias correctamente formateada
cualitativas = datos.select_dtypes(include=['object'])

print("\n++ Tablas de Frecuencia para Variables Cualitativas ++")
for col in cualitativas.columns:
    tabla_frec = datos[col].value_counts().reset_index()
    tabla_frec.columns = [col, "Frecuencia"]
    tabla_frec["Porcentaje"] = (tabla_frec["Frecuencia"] / tabla_frec["Frecuencia"].sum()) * 100
    print(f"\nTabla de Frecuencia - {col}:")
    print(tabla_frec)

# Explicación de los resultados
print("\n++ Explicación de los resultados ++")
print("Para las variables cuantitativas, se ha utilizado la regla empírica para verificar cuántos valores están dentro de 1, 2 y 3 desviaciones estándar de la media.")
print("Si los valores están cerca de 68%, 95% y 99.7%, y la asimetría es cercana a 0, la variable sigue una distribución normal.")
print("Para las variables cualitativas, se generaron tablas de frecuencia que muestran:")
print("  - El valor de cada categoría.")
print("  - Su frecuencia absoluta.")
print("  - Su porcentaje respecto al total.")

# Inciso 4: Análisis de presupuesto, ingresos y votos

print("\n++ Inciso 4: Análisis de presupuesto, ingresos y votos ++")

# 4.1. Las 10 películas con más presupuesto
if "budget" in datos.columns and "title" in datos.columns:
    top_budget = datos[['title', 'budget']].dropna().sort_values(by='budget', ascending=False).head(10)
    print("\n4.1. Las 10 películas con más presupuesto:")
    print(top_budget)
else:
    print("\n4.1. No se encontró la columna 'budget' en los datos.")

# 4.2. Las 10 películas con más ingresos
if "revenue" in datos.columns and "title" in datos.columns:
    top_revenue = datos[['title', 'revenue']].dropna().sort_values(by='revenue', ascending=False).head(10)
    print("\n4.2. Las 10 películas con más ingresos:")
    print(top_revenue)
else:
    print("\n4.2. No se encontró la columna 'revenue' en los datos.")

# 4.3. La película con más votos
if "voteCount" in datos.columns and "title" in datos.columns:
    top_votes = datos[['title', 'voteCount']].dropna().sort_values(by='voteCount', ascending=False).head(1)
    print("\n4.3. La película con más votos:")
    print(top_votes)
else:
    print("\n4.3. No se encontró la columna 'voteCount' en los datos.")

# Inciso 4.4: La peor película según los votos de los usuarios

print("\n++ 4.4. La peor película según los votos de los usuarios ++")

# Verificar si las columnas necesarias existen
if "voteAvg" in datos.columns and "title" in datos.columns:
    worst_movie = datos[['title', 'voteAvg']].dropna().sort_values(by='voteAvg', ascending=True).head(1)
    print("\nLa peor película según el promedio de votos:")
    print(worst_movie)
else:
    print("\nNo se encontró la columna 'voteAvg' en los datos.")

# Inciso 4.5: Cantidad de películas por año y gráfico de barras

import matplotlib.pyplot as plt

# Inciso 4.5: Cantidad de películas por año y gráfico de barras

import matplotlib.pyplot as plt

print("\n++ 4.5. Cantidad de películas por año y año con más películas ++")

# Verificar si la columna 'releaseDate' existe
if "releaseDate" in datos.columns:
    # Convertir la columna a tipo fecha (por si no lo está) y extraer solo el año
    datos["releaseYear"] = pd.to_datetime(datos["releaseDate"], errors="coerce").dt.year

    # Contar cuántas películas se hicieron en cada año
    peliculas_por_anio = datos["releaseYear"].value_counts().sort_index()

    # Mostrar los resultados
    print("\nCantidad de películas por año:")
    print(peliculas_por_anio)

    # Encontrar el año con más películas
    anio_mas_peliculas = peliculas_por_anio.idxmax()
    cantidad_maxima = peliculas_por_anio.max()

    print(f"\nEl año con más películas fue {anio_mas_peliculas} con {cantidad_maxima} películas.")

    # Crear el gráfico de barras
    plt.figure(figsize=(12, 6))
    plt.bar(peliculas_por_anio.index, peliculas_por_anio.values, color="skyblue")
    plt.xlabel("Año")
    plt.ylabel("Cantidad de Películas")
    plt.title("Número de películas por año")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

else:
    print("\nNo se encontró la columna 'releaseDate' en los datos.")

#Ifs usados principalmente para ver si hay error en las variables
