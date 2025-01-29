"""
Entrega 1 Analisis explorativo
Abby Donis 22440import numpy as np
import matplotlib.pyplot as plt
"""
import pandas as pd
import zipfile

#Lee zip
with zipfile.ZipFile('movies.zip', 'r') as z:
    with z.open('movies.csv') as f:
        datos = pd.read_csv(f, encoding='ISO-8859-1')
#delimitacion a 3 decimales
pd.set_option('display.float_format', '{:.3f}'.format)

#Inciso 1
"""
print(datos.describe())
"""


#Inciso 2 (col = columan)
"""
Si la variable:
Es tipo objeto = cualitiativa normal u ordinal
No. finito de categorías = ordinal
-----------------
Es numerica = se revisa si es conitnua o discreta (revisar si el no de valores unicos es grande o pequeño)
"""
"""
print('++ Clasificación tipos de variables ++')
print('.......................................')
def variable_type(col):
    if col.dtype == 'object':
       return 'Cualitativa Nominal u Ordinal'
    elif col.dtype in ['int64', 'float64']:
        if len(col.unique()) > 20:
            return 'Cualitativa Continua'
        else:
            return 'Cualitativa Discreta'
    else:
        'No se pudo identificar'
for col in datos.columns:
    tipo = variable_type(datos[col])
    print (f'Columna: {col} --- Tipo: {tipo}')
"""