import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as km
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np


datos = pd.read_csv("movies.csv", encoding='ISO-8859-1')

# 1.4
#No incluir valores no numericos y quitar NA

datos= datos.dropna(subset=['budget', 'revenue', 'runtime', 'popularity', 'voteAvg', 'voteCount',
                'genresAmount', 'productionCoAmount', 'productionCountriesAmount', 'actorsAmount',
                'castWomenAmount', 'castMenAmount'])
k_col_datos =['budget', 'revenue', 'runtime', 'popularity', 'voteAvg', 'voteCount',
                'genresAmount', 'productionCoAmount', 'productionCountriesAmount', 'actorsAmount',
                'castWomenAmount', 'castMenAmount']

X = datos[k_col_datos] #extracción de coljumnas

#normalizacion de datos (evitar sesgos por diferentes escalas)
norm = StandardScaler()
X_norm = norm.fit_transform(X)
#recordar que el numero de custars es 6
"""
K-means
"""
kmeans = km(n_clusters=6, random_state=42)
datos['clusterKMeans'] = kmeans.fit_predict(X_norm)
#nuevo csv
datos_col_toNewCSV = ['title'] + k_col_datos +['clusterKMeans'] if 'title' in datos.columns else k_col_datos + ['clusterKmeans']
datos[datos_col_toNewCSV].to_csv("movies_KMeans.csv", index=False)

print(datos[['title', 'clusterKMeans']].head())

"""
Clustering jerarquico
"""
clusteringJ=AgglomerativeClustering(n_clusters=6, metric='euclidean', linkage='ward')
datos['clusterJerarquico'] = clusteringJ.fit_predict(X_norm)

datos_col_toNewCSV = ['title'] + k_col_datos +['clusterJerarquico'] if 'title' in datos.columns else k_col_datos + ['clusterJerarquico']
datos[datos_col_toNewCSV].to_csv("movies_ClusterJerarquico.csv", index=False)
#confirmacion
print(datos[["title", "clusterJerarquico"]].head() if "title" in datos.columns else datos[["clusterJerarquico"]].head())

"""
Comparacion
"""
kmeans_count = np.bincount(datos['clusterKMeans'])
jerarquia_counts = np.bincount(datos['clusterJerarquico'])
#histogramas
fig, axes = plt.subplots(1,2, figsize=(12,5))
#kmean
axes[0].bar(range(len(kmeans_count)), kmeans_count, color='red', alpha=0.7)
axes[0].set_title('Distribucion Clusters - KMeans')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('No. películas')
#Jerarquico
axes[1].bar(range(len(jerarquia_counts)), jerarquia_counts, color='purple', alpha=0.7)
axes[1].set_title('Distribucion Clusters - Jerarquico')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('No. películas')

plt.show()