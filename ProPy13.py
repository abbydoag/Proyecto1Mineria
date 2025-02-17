import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset
file_path = "movies.csv"
movies_df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Eliminar columnas irrelevantes para clustering
columns_to_drop = ["id", "homePage", "originalTitle", "title", "releaseDate", "video"]
movies_cleaned = movies_df.drop(columns=columns_to_drop)

# Manejar valores nulos: reemplazar por la media en columnas numéricas
numerical_columns = movies_cleaned.select_dtypes(include=["float64", "int64"]).columns
movies_cleaned[numerical_columns] = movies_cleaned[numerical_columns].fillna(movies_cleaned[numerical_columns].mean())

# Seleccionar solo columnas numéricas para clustering
numerical_data = movies_cleaned.select_dtypes(include=["float64", "int64"]).values

# Normalizar los datos manualmente (media 0, desviación estándar 1)
mean_vals = np.mean(numerical_data, axis=0)
std_vals = np.std(numerical_data, axis=0)
scaled_data = (numerical_data - mean_vals) / std_vals  # <-- Aquí se asegura que scaled_data esté definido

# Función para aplicar k-means manualmente y calcular inercia
def kmeans_inertia(X, k, max_iters=100):
    np.random.seed(42)
    n_samples, n_features = X.shape

    # Inicializar centros aleatorios
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iters):
        # Asignar cada punto al centroide más cercano
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Recalcular centroides
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Si los centroides no cambian, detener iteración
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Calcular inercia (suma de distancias al centroide más cercano)
    inertia = np.sum((X - centroids[labels]) ** 2)
    return inertia

# Calcular la inercia para diferentes valores de k
k_values = range(1, 11)
inertia_values = [kmeans_inertia(scaled_data, k) for k in k_values]

# Imprimir los valores de inercia para cada k
for k, inertia in zip(k_values, inertia_values):
    print(f"k={k}, Inercia={inertia}")

# Graficar la curva de codo manualmente con Numpy
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia_values, marker="o", linestyle="-")
plt.xlabel("Número de Clústeres (k)")
plt.ylabel("Inercia")
plt.title("Método del Codo para determinar k óptimo")
plt.xticks(k_values)
plt.grid()
plt.show()
