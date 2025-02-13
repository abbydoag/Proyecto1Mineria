import pandas as pd

datos = pd.read_csv("movies.csv", encoding='ISO-8859-1')
pd.set_option('display.float_format', '{:.3f}'.format)

#Inciso 1
print(datos.head())