"""
Entrega 1 Analisis explorativo
Abby Donis 22440
Wilson Calderón 22018
"""
import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Lee zip
with zipfile.ZipFile('Movies.zip', 'r') as z:
    with z.open('Movies.csv') as f:
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

# Inciso 4.6: Análisis de géneros en películas recientes, predominantes y más largas
print("\n++ 4.6. Análisis de Géneros ++")

# Asegurar que la columna releaseDate sea reconocida como fecha
if "releaseDate" in datos.columns and "genres" in datos.columns:
    datos["releaseDate"] = pd.to_datetime(datos["releaseDate"], errors="coerce")
    
    # Obtener las 20 películas más recientes
    peliculas_recientes = datos.sort_values(by="releaseDate", ascending=False).head(20)
    
    # Extraer el primer género de cada película (separado por "|")
    peliculas_recientes["genero_principal"] = peliculas_recientes["genres"].str.split("|").str[0]
    
    print("\nGéneros principales de las 20 películas más recientes:")
    print(peliculas_recientes[["title", "genero_principal"]])

    # Género principal más frecuente en todo el dataset
    datos["genero_principal"] = datos["genres"].str.split("|").str[0]
    genero_predominante = datos["genero_principal"].value_counts()
    
    print("\nGénero principal que predomina en el conjunto de datos:")
    print(genero_predominante)

    # Gráfico de barras de los géneros predominantes
    plt.figure(figsize=(12, 6))
    genero_predominante.head(10).plot(kind="bar", color="skyblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Género")
    plt.ylabel("Cantidad de Películas")
    plt.title("Géneros Predominantes en el Dataset")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Películas más largas (top 10 por duración)
    if "runtime" in datos.columns:
        peliculas_largas = datos.sort_values(by="runtime", ascending=False).head(10)
        peliculas_largas["genero_principal"] = peliculas_largas["genres"].str.split("|").str[0]

        print("\nGéneros principales de las películas más largas:")
        print(peliculas_largas[["title", "runtime", "genero_principal"]])
    
else:
    print("\nNo se encontraron las columnas necesarias ('releaseDate' y 'genres') en los datos.")

# Inciso 4.7: Género principal con mayores ganancias

print("\n++ 4.7. Género principal con mayores ganancias ++")

# Verificar si las columnas necesarias existen
if "genres" in datos.columns and "revenue" in datos.columns and "budget" in datos.columns:
    # Calcular las ganancias (revenue - budget)
    datos["ganancias"] = datos["revenue"] - datos["budget"]

    # Extraer el primer género de cada película
    datos["genero_principal"] = datos["genres"].str.split("|").str[0]

    # Calcular la ganancia total por género
    ganancias_por_genero = datos.groupby("genero_principal")["ganancias"].sum().sort_values(ascending=False)

    print("\nGanancias totales por género principal:")
    print(ganancias_por_genero)

    # Género con mayores ganancias
    genero_mayor_ganancia = ganancias_por_genero.idxmax()
    mayor_ganancia = ganancias_por_genero.max()

    print(f"\nEl género con mayores ganancias es '{genero_mayor_ganancia}' con un total de {mayor_ganancia:,.2f} en ganancias.")

    # Graficar las ganancias por género
    plt.figure(figsize=(12, 6))
    ganancias_por_genero.head(10).plot(kind="bar", color="skyblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Género")
    plt.ylabel("Ganancias Totales")
    plt.title("Géneros con Mayores Ganancias")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

else:
    print("\nNo se encontraron las columnas necesarias ('genres', 'revenue', 'budget') en los datos.")

# Inciso 4.8: Influencia de la cantidad de actores en los ingresos y evolución en el tiempo

print("\n++ 4.8. Análisis de la cantidad de actores y los ingresos ++")

# Filtrar valores extremos y asegurarse de que los datos sean numéricos
if "actorsAmount" in datos.columns and "revenue" in datos.columns and "releaseDate" in datos.columns:
    # Convertir a numérico, eliminando errores y valores nulos
    datos["actorsAmount"] = pd.to_numeric(datos["actorsAmount"], errors="coerce")
    datos["revenue"] = pd.to_numeric(datos["revenue"], errors="coerce")

    # Filtrar datos con actores dentro de un rango lógico (ejemplo: 1 a 500)
    datos_filtrados = datos[(datos["actorsAmount"] > 0) & (datos["actorsAmount"] <= 500)]

    # Relación entre la cantidad de actores y los ingresos
    plt.figure(figsize=(10, 5))
    plt.scatter(datos_filtrados["actorsAmount"], datos_filtrados["revenue"], alpha=0.5, color="blue")
    plt.xlabel("Cantidad de Actores")
    plt.ylabel("Ingresos ($)")
    plt.title("Relación entre la cantidad de actores y los ingresos (Valores Filtrados)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    print("\nInterpretación:")
    print("Se han filtrado valores extremos (> 500 actores) para mejorar la visualización.")
    print("Si los puntos muestran una tendencia ascendente, significa que más actores podrían influir en mayores ingresos.")
    print("Si no hay un patrón claro, significa que la cantidad de actores no tiene un impacto significativo en los ingresos.")

else:
    print("\nNo se encontraron las columnas necesarias ('actorsAmount', 'revenue' y 'releaseDate') en los datos.")
#Ifs usados principalmente para ver si hay error en las variables


# Inciso 4.9
correlacion = datos[['revenue', 'popularity', 'castWomenAmount', 'castMenAmount']].corr()
print(correlacion)

plt.figure(figsize=(12,10))
#revenue-castWomenAmount
plt.subplot(2, 2, 1)
plt.scatter(datos['castWomenAmount'], datos['revenue'],alpha=0.5, c='green')
plt.title('Relación entre ingresos y cantidad de actrices')
plt.xlabel('Cantidad Actrices')
plt.ylabel('Ingresos')
#revenue-castMenAmount
plt.subplot(2, 2, 2)
plt.scatter(datos['castMenAmount'], datos['revenue'],alpha=0.5, c='green')
plt.title('Relación entre ingresos y cantidad de actores')
plt.xlabel('Cantidad actores')
plt.ylabel('Ingresos')
#popularidad-castWomenAmount
plt.subplot(2, 2, 3)
plt.scatter(datos['castWomenAmount'], datos['popularity'],alpha=0.5, c='blue')
plt.title('Relación entre popularidad y cantidad de Actrices')
plt.xlabel('Cantidad Actrices')
plt.ylabel('Popularidad')
#popularidad-castMenAmount
plt.subplot(2, 2, 4)
plt.scatter(datos['castMenAmount'], datos['popularity'],alpha=0.5, c='blue')
plt.title('Relación entre popularidad y cantidad de actores')
plt.xlabel('Cantidad Actores')
plt.ylabel('Popularidad')

plt.tight_layout()
plt.show()


#Inciso 4.10
datos['voteAvg'] = pd.to_numeric(datos['voteAvg'], errors='coerce')
datos = datos[datos['director'].notna()] #Descartar NaN
sort_movies = datos.sort_values(by = 'voteAvg', ascending=False) #mayor a menor
top20_movies = sort_movies.head(20)
top20_movies_directors = top20_movies['director']
#dataframe
top20_movies_and_directors = datos[datos['director'].isin(top20_movies_directors)]
top20 = top20_movies_and_directors.sort_values(by='voteAvg', ascending=False)

print('TOP 20 PELICULAS Y  SUS DIRECTORES')
print(top20[['director', 'title', 'voteAvg']])


#Inciso 4.11
datos = datos.dropna(subset=['revenue', 'budget'])
correlation_rev_bud = datos[['revenue', 'budget']].corr()
print(correlation_rev_bud)

plt.figure(figsize=(12,6))
#Histograma Presupuesto
plt.subplot(1,2,1)
plt.hist(datos['budget'], bins = 30, color='red', alpha=0.7)
plt.title('Distribución de presupuestos')
plt.xlabel('Presupuesto')
plt.ylabel('Frecuencia')
#Histograma Ingresos
plt.subplot(1,2,2)
plt.hist(datos['revenue'], bins=30, color='blue', alpha=0.7)
plt.title('Distobución de ingresos')
plt.xlabel('Ingresos')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

#Correlacion (scatter plot) ingresos vs presupuesto
plt.figure(figsize=(8,6))
plt.scatter(datos['budget'], datos['revenue'], color='green', alpha=0.5)
plt.title('Relacion entre ingresos y presupuestos')
plt.xlabel('Presupuesto')
plt.ylabel('Ingreso')
plt.show()


#Incisos 4.12 y 4.13
datos['releaseDate'] = pd.to_datetime(datos['releaseDate'], errors='coerce')
datos = datos.dropna(subset=['revenue','releaseDate'])
datos['month'] = datos['releaseDate'].dt.month #mes
month_rev = datos.groupby('month')['revenue'].mean()
print('Promedio ingresos al mes')
print(month_rev)
#grafico
plt.figure(figsize=(10, 6))
month_rev.plot(kind='bar', color='red')
plt.title('Ingreso promedio (Mes)')
plt.xlabel('Mes')
plt.ylabel('Ingreso (Promedio)')
plt.xticks(rotation=0) #orientacion meses
plt.show()
#Promedio peliculas/mes
movies_month= datos['month'].value_counts().sort_index()
avg_movesMonth = movies_month.mean()
print(f'Promedio de películas al mes:\n{movies_month}')
plt.figure(figsize=(10, 6))
movies_month.plot(kind='bar', color='lightcoral')
plt.title('Número de peliculas estrenadas al mes')
plt.xlabel('Mes')
plt.ylabel('No. de Películas')
plt.xticks(rotation=0)
plt.show()

#Inciso 4.14 avgVotes, Rvenue
datos = datos.dropna(subset=['voteAvg', 'revenue'])
correlation_avgVote_revenue = datos[['voteAvg', 'revenue']].corr()
print(correlation_avgVote_revenue)
#scatterplot
plt.figure(figsize=(8,6))
plt.scatter(datos['voteAvg'], datos['revenue'], color='green', alpha=0.6)
plt.title('Relacion calificaciones e ingresos')
plt.xlabel('Calificacion')
plt.ylabel('Ingresos')
plt.show()

#Inciso 4.15
datos['video'] = datos['video'].replace({'TRUE':True, 'FALSE':False, 'NA': pd.NA}) #conv a bool
datos['homePage'] = datos['homePage'].replace('NA', pd.NA) #NaN para filtro
#Video
video = datos[datos['video']==True]
no_video = datos[datos['video']==False]
avg_rev_video = video['revenue'].mean()
avg_rev_noVideo = no_video['revenue'].mean()
print('Películas con videos:', len(video))
print('Películas sin videos:', len(no_video))
print('Promedio ingresos con videos:', avg_rev_video)
print('Promedio ingresos sin videos:', avg_rev_noVideo)
print ('------------------------------------------------')
#Home page
homePage = datos[datos['homePage'].notna()]
no_HomePage = datos[datos['homePage'].isna()]
avg_rev_homePage = homePage['revenue'].mean()
avg_rev_noHomePage = no_HomePage['revenue'].mean()
print('Películas con pagina web:', len(homePage))
print('Películas sin pagina web:', len(no_HomePage))
print('Promedio ingresos con pagina web:', avg_rev_homePage)
print('Promedio ingresos sin pagina web:', avg_rev_noHomePage)
#comparacion video ingreso
plt.figure(figsize=(10,6))
plt.bar(['Videos', 'Sin Videos'], [avg_rev_video, avg_rev_noVideo], color=['red', 'blue'])
plt.title('Promedio de ingresos de peliculas con y sin videos promocionales')
plt.ylabel('Ingreso promedio')
plt.show()
#comparacion pagina web ingreso
plt.figure(figsize=(10,6))
plt.bar(['Pagina Web', 'Sin Pagina Web'], [avg_rev_homePage, avg_rev_noHomePage], color=['green', 'purple'])
plt.title('Promedio de ingresos de peliculas con y sin paginas web')
plt.ylabel('Ingreso promedio')
plt.show()


#Inciso 4.16
datos['actorsPopularity'] = pd.to_numeric(datos['actorsPopularity'], errors='coerce')
corr_actorPopularity_rev = datos[['actorsPopularity', 'revenue']].corr()
print(corr_actorPopularity_rev)
#scatterplot
plt.figure(figsize=(10,6))
plt.scatter(datos['actorsPopularity'], datos['revenue'], alpha=0.5, color='orange')
plt.title('Relación entre popularidad del elenco e ingresos')
plt.xlabel('Popularidad elenco')
plt.ylabel('Ingresos')
plt.show()

# Inciso 4.17: Duración promedio de las películas por género principal
print("\n++ 4.17. Duración promedio de las películas por género principal ++")

# Verificar si las columnas necesarias existen
if "genres" in datos.columns and "runtime" in datos.columns:
    # Extraer el primer género de cada película
    datos["genero_principal"] = datos["genres"].str.split("|").str[0]

    # Calcular la duración promedio por género
    duracion_por_genero = datos.groupby("genero_principal")["runtime"].mean().sort_values(ascending=False)

    print("\nDuración promedio de las películas por género:")
    print(duracion_por_genero)

    # Graficar la duración promedio de las películas por género
    plt.figure(figsize=(12, 6))
    duracion_por_genero.plot(kind="bar", color="skyblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Género")
    plt.ylabel("Duración Promedio (minutos)")
    plt.title("Duración Promedio de las Películas por Género")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

else:
    print("\nNo se encontraron las columnas necesarias ('genres', 'runtime') en los datos.")

#PUNTOS EXTRA COMIENZAN AQUI

# Inciso 4.18: Relación entre el presupuesto y las calificaciones
print("\n++ 4.18. Relación entre el presupuesto y las calificaciones ++")

# Verificar si las columnas necesarias existen
if "budget" in datos.columns and "voteAvg" in datos.columns:
    # Convertir a numérico para evitar errores
    datos["budget"] = pd.to_numeric(datos["budget"], errors="coerce")
    datos["voteAvg"] = pd.to_numeric(datos["voteAvg"], errors="coerce")

    # Filtrar presupuestos irreales (ejemplo: mayor a $10,000,000,000)
    datos_filtrados = datos[(datos["budget"] > 0) & (datos["budget"] <= 1e10)]

    # Generar gráfico de dispersión entre presupuesto y calificaciones
    plt.figure(figsize=(10, 5))
    plt.scatter(datos_filtrados["budget"], datos_filtrados["voteAvg"], alpha=0.5, color="purple")
    plt.xlabel("Presupuesto ($)")
    plt.ylabel("Calificación Promedio")
    plt.title("Relación entre el Presupuesto y las Calificaciones")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    print("\nInterpretación:")
    print("Si los puntos muestran una tendencia ascendente, significa que las películas con mayor presupuesto reciben mejores calificaciones.")
    print("Si los puntos están dispersos sin un patrón claro, significa que el presupuesto no afecta significativamente las calificaciones.")

else:
    print("\nNo se encontraron las columnas necesarias ('budget', 'voteAvg') en los datos.")

# Inciso 4.19: Productoras con más películas y mayores ingresos

print("\n++ 4.19. Productoras con más películas y mayores ingresos ++")

# Verificar si las columnas necesarias existen
if "productionCompany" in datos.columns and "revenue" in datos.columns:
    # Contar cuántas películas ha producido cada compañía
    productoras_mas_peliculas = datos["productionCompany"].value_counts().head(10)

    print("\nTop 10 productoras con más películas:")
    print(productoras_mas_peliculas)

    # Calcular ingresos totales por productora
    ingresos_por_productora = datos.groupby("productionCompany")["revenue"].sum().sort_values(ascending=False).head(10)

    print("\nTop 10 productoras con mayores ingresos:")
    print(ingresos_por_productora)

    # Graficar las 10 productoras con más películas
    plt.figure(figsize=(12, 6))
    productoras_mas_peliculas.plot(kind="bar", color="orange", edgecolor="black", alpha=0.7)
    plt.xlabel("Productora")
    plt.ylabel("Cantidad de Películas")
    plt.title("Top 10 Productoras con Más Películas")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Graficar las 10 productoras con mayores ingresos
    plt.figure(figsize=(12, 6))
    ingresos_por_productora.plot(kind="bar", color="green", edgecolor="black", alpha=0.7)
    plt.xlabel("Productora")
    plt.ylabel("Ingresos Totales ($)")
    plt.title("Top 10 Productoras con Mayores Ingresos")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

else:
    print("\nNo se encontraron las columnas necesarias ('productionCompany', 'revenue') en los datos.")
#Ifs usados principalmente para ver si hay error en las variables

#Inciso 5.4 
datos['productionCountry'] = datos['productionCountry'].str.strip()
datos['revenue'] = pd.to_numeric(datos['revenue'], errors='coerce')
datos['productionCountry'] = datos['productionCountry'].fillna('')
#division |
datos['productionCountry'] = datos['productionCountry'].apply(
    lambda x: x.split('|') if isinstance(x, str) else [])
#Peliculas/pais
countries = [country for sublist in datos['productionCountry'] for country in sublist]
num_movies_country = pd.Series(countries).value_counts()
#Top 5
top_5 = num_movies_country.head(5)
top_5_countries = top_5.index.tolist()

top_countries = datos[datos['productionCountry'].apply(
    lambda x: any(country in top_5_countries for country in x))]
#Ingresos top 5
rev_country = {}
for _, row in top_countries.iterrows():
    countries = row['productionCountry']
    revenue = row['revenue']
    for country in countries:
        if country in top_5_countries:
            if country not in rev_country:
                rev_country[country] = {'total_revenue': 0, 'count': 0}
            rev_country[country]['total_revenue'] += revenue
            rev_country[country]['count'] += 1
avg_rev_country = {country: data['total_revenue'] / data['count'] for country, data in rev_country.items()}
avg_rev = pd.DataFrame(list(avg_rev_country.items()), columns=['Pais', 'Ingreso promedio'])
avg_rev = avg_rev.sort_values(by='Ingreso promedio', ascending=False)

print(' Top 5 paises con mas producciones e ingresos')
print('+---------------------------------------------+')
print(top_5)
print(avg_rev)

#Inciso 5.5
datos['genres'] = datos['genres'].str.strip()
datos['genres'] = datos['genres'].fillna('')
datos['genres'] = datos['genres'].apply(
    lambda x: x.split('|') if isinstance(x, str) else [])
#genero
all_genres=[genre for sublist in datos['genres'] for genre in sublist]
genresDF = pd.DataFrame(all_genres, columns=['genre'])
#popularidad
genresDF['popularity'] = datos['popularity'].repeat(datos['genres'].apply(len)).reset_index(drop=True)
avg_popularity_genre = genresDF.groupby('genre')['popularity'].mean().sort_values(ascending=False)
print('Popularidad promedio por genero')
print(avg_popularity_genre)
#grafica
plt.figure(figsize=(12,6))
avg_popularity_genre.plot(kind='bar', color='blue')
plt.title('Popularidad promedio por genero')
plt.xlabel('Genero')
plt.ylabel('Popularidad (Promedio)')
plt.xticks(rotation=45) #orientacion meses
plt.show()

pop_genre = [genresDF[genresDF['genre'] == genre]['popularity'].values for genre in avg_popularity_genre.index]
plt.boxplot(pop_genre, vert=False)
plt.yticks(range(1, len(avg_popularity_genre)+1), avg_popularity_genre.index)
plt.xlabel('Popularida (Promedio)')
plt.ylabel('Genero')
plt.title('Distribucion popularidad por genero')
plt.show()


#Inciso 5.6
datos['runtime'] = pd.to_numeric(datos['runtime'], errors='coerce')
datos['revenue'] = pd.to_numeric(datos['revenue'], errors='coerce')
#correlacion
runtime_rev = datos[['runtime', 'revenue']].corr()
print(runtime_rev)
plt.figure(figsize=(10, 6))
plt.scatter(datos['runtime'], datos['revenue'], alpha=0.5, color='blue')

plt.title('Relación entre Duración e Ingresos')
plt.xlabel('Duración (minutos)')
plt.ylabel('Ingresos')
plt.show()