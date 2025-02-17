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
scaled_data = (numerical_data - mean_vals) / std_vals

# Función detallada para calcular el estadístico de Hopkins con más procedimiento
def hopkins_statistic(X, sample_ratio=0.1):
    """
    Calcula el estadístico de Hopkins para medir la tendencia al agrupamiento.
    Muestra más detalles del cálculo, incluyendo las distancias reales y sintéticas.
    """
    n_samples = X.shape[0]
    n_selected = max(1, int(sample_ratio * n_samples))  # Seleccionar 10% de los datos

    # Seleccionar puntos aleatorios del dataset
    np.random.seed(42)
    random_indices = np.random.choice(n_samples, n_selected, replace=False)
    random_points = X[random_indices]

    # Generar puntos sintéticos dentro del mismo rango de valores
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    synthetic_points = np.random.uniform(min_vals, max_vals, (n_selected, X.shape[1]))

    # Calcular la distancia mínima entre cada punto y el resto del dataset
    def min_distances(points, dataset):
        """
        Calcula la distancia mínima entre cada punto en 'points' y todos los puntos en 'dataset'.
        """
        distances = []
        for p in points:
            dist = np.linalg.norm(dataset - p, axis=1)  # Distancia Euclidiana
            min_dist = np.min(dist[dist > 0])  # Ignorar la distancia cero (sí mismo)
            distances.append(min_dist)
        return np.array(distances)

    # Distancias a los vecinos más cercanos en el dataset real y sintético
    real_distances = min_distances(random_points, X)
    synthetic_distances = min_distances(synthetic_points, X)

    # Calcular el estadístico de Hopkins
    U = np.sum(synthetic_distances)  # Suma de distancias sintéticas
    W = np.sum(real_distances)  # Suma de distancias reales
    hopkins_value = U / (U + W)

    # Mostrar detalles del procedimiento
    print(f"\n--- Estadístico de Hopkins ---")
    print(f"Número de muestras seleccionadas: {n_selected}")
    print(f"Suma de distancias sintéticas (U): {U:.4f}")
    print(f"Suma de distancias reales (W): {W:.4f}")
    print(f"Estadístico de Hopkins: {hopkins_value:.4f}\n")

    # Visualización de las distancias
    plt.figure(figsize=(8, 5))
    plt.hist(real_distances, bins=20, alpha=0.6, label="Distancias Reales", color="blue")
    plt.hist(synthetic_distances, bins=20, alpha=0.6, label="Distancias Sintéticas", color="red")
    plt.xlabel("Distancia")
    plt.ylabel("Frecuencia")
    plt.title("Comparación de Distancias en el Cálculo de Hopkins")
    plt.legend()
    plt.grid()
    plt.show()

    return hopkins_value

# Calcular el estadístico de Hopkins con detalles
hopkins_value = hopkins_statistic(scaled_data)
