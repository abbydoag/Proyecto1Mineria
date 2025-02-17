import pandas as pd
import numpy as np

# Cargar el dataset
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

# Calcular la matriz de covarianza
cov_matrix = scaled_pca_data.cov()

# Calcular los autovalores y autovectores
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Ordenar autovalores y autovectores de mayor a menor
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Seleccionar los primeros 5 componentes principales
num_components = 5
principal_components = scaled_pca_data.dot(eigenvectors[:, :num_components])

# Convertir a DataFrame
pca_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(num_components)])

# Mostrar la varianza explicada por cada componente
explained_variance = eigenvalues / eigenvalues.sum()

# Imprimir resultados
print("Varianza explicada por cada componente:")
for i in range(num_components):
    print(f"PC{i+1}: {explained_variance[i]:.4f}")

print("\nPrimeras filas de los componentes principales:")
print(pca_df.head())
