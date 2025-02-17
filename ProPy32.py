import pandas as pd
import numpy as np

# Cargar el dataset nuevamente
file_path = "movies.csv"
movies_df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Seleccionar variables numéricas para PCA
numerical_columns = ["budget", "revenue", "runtime", "popularity", "voteAvg", "voteCount",
                     "genresAmount", "productionCoAmount", "productionCountriesAmount",
                     "actorsAmount", "castWomenAmount", "castMenAmount"]

# Filtrar solo las columnas numéricas y manejar valores nulos con la media
pca_data = movies_df[numerical_columns].fillna(movies_df[numerical_columns].mean())

# Normalizar los datos (media 0, desviación estándar 1) evitando división por 0
mean_vals = pca_data.mean()
std_vals = pca_data.std()
std_vals[std_vals == 0] = 1  # Evitar errores de división por cero
scaled_pca_data = (pca_data - mean_vals) / std_vals

# Calcular la matriz de correlación
correlation_matrix = scaled_pca_data.corr().values

# Cálculo del índice KMO
def calculate_kmo(corr_matrix):
    inv_corr_matrix = np.linalg.pinv(corr_matrix)  # Matriz inversa pseudo-inversa
    n = corr_matrix.shape[0]
    partial_corr_sum = np.sum((1 - np.diag(inv_corr_matrix))**2)
    total_corr_sum = np.sum(corr_matrix**2)
    kmo_value = total_corr_sum / (total_corr_sum + partial_corr_sum)
    return kmo_value

# Test de esfericidad de Bartlett
def bartlett_test(corr_matrix, n_samples):
    det_corr_matrix = np.linalg.det(corr_matrix)
    chi_square_stat = -((n_samples - 1) - (2 * (corr_matrix.shape[0] + 1) / 3)) * np.log(det_corr_matrix)
    dof = (corr_matrix.shape[0] * (corr_matrix.shape[0] - 1)) / 2  # Grados de libertad
    p_value = np.exp(-chi_square_stat / 2)  # Aproximación para el p-valor
    return chi_square_stat, p_value

# Calcular KMO
kmo_value = calculate_kmo(correlation_matrix)

# Calcular Bartlett
n_samples = pca_data.shape[0]
chi_square_stat, p_value = bartlett_test(correlation_matrix, n_samples)

# Mostrar resultados
kmo_value, chi_square_stat, p_value

# Mostrar resultados en un formato claro
print("Índice KMO:", round(kmo_value, 4))

print("\nTest de esfericidad de Bartlett:")
print("Chi-cuadrado:", round(chi_square_stat, 2))
print("p-valor:", round(p_value, 6))  # Se usa 6 decimales para mostrar valores pequeños correctamente
