import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as km
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib.cm as cm

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
#datos_col_toNewCSV = ['title'] + k_col_datos +['clusterKMeans'] if 'title' in datos.columns else k_col_datos + ['clusterKmeans']
#datos[datos_col_toNewCSV].to_csv("movies_KMeans.csv", index=False)

#print(datos[['title', 'clusterKMeans']].head())

"""
Clustering jerarquico
"""
clusteringJ=AgglomerativeClustering(n_clusters=6, metric='euclidean', linkage='ward')
datos['clusterJerarquico'] = clusteringJ.fit_predict(X_norm)

#datos_col_toNewCSV = ['title'] + k_col_datos +['clusterJerarquico'] if 'title' in datos.columns else k_col_datos + ['clusterJerarquico']
#datos[datos_col_toNewCSV].to_csv("movies_ClusterJerarquico.csv", index=False)
#confirmacion
#print(datos[["title", "clusterJerarquico"]].head() if "title" in datos.columns else datos[["clusterJerarquico"]].head())

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


#1.5
#silhoulette kmeans
silhouette_kMean = silhouette_score(X_norm, datos['clusterKMeans'])
silhouette_kMean_sample = silhouette_samples(X_norm, datos['clusterKMeans'])
fig, ax = plt.subplots(figsize=(8,6))
y_lower = 10
for i in range (len(set(datos['clusterKMeans']))):
    ith_cluster_silhouette_values = silhouette_kMean_sample[datos['clusterKMeans']== i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i)/len(set(datos['clusterKMeans'])))
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor= color, 
                     edgecolor= color, alpha= 0.7)
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
ax.set_title("Silhouette - KMeans")
ax.set_xlabel("Coeficiente silueta")
ax.set_ylabel("Cluster")
#promedio
ax.axvline(x=silhouette_kMean, color="gold", linestyle="--")
ax.set_yticks([])  
plt.show()

#silhoulette jerarquico
silhouette_J = silhouette_score(X_norm, datos['clusterJerarquico'])
silhouette_J_sample = silhouette_samples(X_norm, datos['clusterJerarquico'])
fig, ax = plt.subplots(figsize=(8,6))
y_lower = 10
for i in range (len(set(datos['clusterJerarquico']))):
    ith_cluster_silhouette_values = silhouette_J_sample[datos['clusterJerarquico']== i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i)/len(set(datos['clusterJerarquico'])))
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor= color, 
                     edgecolor= color, alpha= 0.7)
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
ax.set_title("Silhouette - Jerarquico")
ax.set_xlabel("Coeficiente silueta")
ax.set_ylabel("Cluster")
#promedio
ax.axvline(x=silhouette_J, color="orange", linestyle=":")
ax.set_yticks([])  
plt.show()

print(f'Silhoutte Score - KMeans: {silhouette_kMean:.4f}')
print(f'Silhoutte Score - Jerarquico: {silhouette_J:.4f}')


#Medidas tendencia central
col_num = datos.select_dtypes(include=[np.number]).columns.tolist()
col_num.remove('id')
col_num.remove('clusterKMeans')
col_num.remove('clusterJerarquico')

media_porClusterK = datos.groupby('clusterKMeans')[col_num].mean()
mediana_porClusterK = datos.groupby('clusterKMeans')[col_num].median()

#Se ve mejor en csv
media_porClusterK.to_csv('media_por_cluster_KMeans.csv')
mediana_porClusterK.to_csv('mediana_por_cluster_KMeans.csv')
print('Ya puede revisar los archivos')