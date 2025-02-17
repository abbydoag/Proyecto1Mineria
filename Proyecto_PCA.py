import inspect
import pandas as pd
from apyori import apriori
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer

datos = pd.read_csv("movies.csv", encoding='ISO-8859-1').dropna()
pd.set_option('display.float_format', '{:.3f}'.format)
#print(datos.head())
#print(datos.columns)
datos = datos.drop(columns=['id','originalTitle','title','homePage'])
#discretizacion
sel_num = make_column_selector(dtype_exclude=object)
sel_cat = make_column_selector(dtype_include=object)
numericas = sel_num(datos)
categoricas = sel_cat(datos)
print(f"Variables numéricas: {numericas}")
print(f"\nVariables categóricas: {categoricas}")


discret_numerico = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
datos[numericas] = discret_numerico.fit_transform(datos[numericas])
#etiquetnado numericas para mas organizacion
labels = ['low', 'medium', 'high', 'veryHigh']
for col in numericas:
    datos[col] = datos[col].map(lambda x: labels[int(x)])

print("Datos discretizados (numéricos):")
print(datos[numericas].head())

#seleccion para reglas
datos_sele = datos[categoricas+numericas]
#conve en lista de valores recorrendo
records = datos_sele.apply(lambda x: list(x.dropna().astype(str)), axis=1).to_list()

reglas_asociacion = apriori(records,min_support=0.2, min_confidence=0.7)
reglas = list(reglas_asociacion)
print(f'\nSe han encontrado {len(reglas)} reglas de asociacion')

#Extracion
izq_regla=[]
der_regla=[]
support = []
confidence = []

for regla in reglas:
    for result in regla.ordered_statistics:
        izq_regla.append(tuple(result.items_base))
        der_regla.append(tuple(result.items_add))
        support.append(regla.support)
        confidence.append(result.confidence)

#DF
print('----------------------------------------------------------------------------------------')
df = pd.DataFrame({
    'ladoIzq': izq_regla,
    'ladoDer': der_regla,
    'support': support,
    'confidence': confidence
})
#filtro reglas
df = df[~df['ladoIzq'].apply(lambda x: all(isinstance(i, int) for i in x))]
df = df[~df['ladoDer'].apply(lambda x: all(isinstance(i, int) for i in x))]
print("Reglas de asociación generadas:\n")
print(df.head())