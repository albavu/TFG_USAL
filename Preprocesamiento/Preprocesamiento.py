#############################################################################
            # Autor: Alba Vallejo Urruchi
            # TFG Ingeniería Informática 2020-2021
            # Universidad de Salamanca
#############################################################################

#!/usr/bin/env python
# coding: utf-8

# ## Preprocesamiento Secuencias DNA 

# Contiene todas las cadenas de DNA que posiciona nucleosomas
# Archivo FASTA que los contiene: '/lhome/ext/usal053/usal0531/TFG/data/Nuc_SP3.fasta'

import re
import pandas as pd
import numpy as np

#Variables
carpeta='/Users/alba_vu/Desktop/curso20-21/TFG/data/input'
#carpeta_s = '/lhome/ext/usal053/usal0531/TFG/data/datosEntrada'
carpeta_salida = '/Users/alba_vu/Desktop/curso20-21/TFG/data/output'



# ## Lectura de datos
## Primero leemos el archivo de csv creado con DataSet_Edition.py
##No-nucleosomas
df = pd.read_csv(carpeta+'/PombeSeq3.csv')
##En caso del servidor
#df = pd.read_csv('/lhome/ext/usal053/usal0531/TFG/data/PombeSeq3.csv')
df.rename(columns = {'0':'secuencia'}, inplace = True)

#Lectura de las secuencias de NDR
df1 = pd.read_csv(carpeta+'/NDRseq2.csv')
#df1 = pd.read_csv('/lhome/ext/usal053/usal0531/TFG/data/NDRseq2.csv')
df = pd.concat([df, df1], axis=0)

#Nucleosomas
n = pd.read_csv(carpeta+'/NucleocSeq.csv')

def string_to_array(cadena):
    cadena = cadena.lower()
    cadena = re.sub('[^acgt]','z',cadena)
    cadena = np.array(list(cadena))
    return cadena

# ## Primera forma
##Los datos no pueden estar en formato alfanumérico, por eso, se pasan a formato numérico para poderlos meter al modelo

# Tipo de cambio:
# 
# N -> 0
# 
# A -> 1
# 
# C -> 2
# 
# T -> 3
# 
# G -> 4


df['secuencia'] = df['secuencia'].apply(lambda x: re.sub(r'([A-Z])(?!$)', r'\1,', x))

df = df.reset_index(drop=True)

n['secuencia'] = n['secuencia'].apply(lambda x: re.sub(r'([A-Z])(?!$)', r'\1,', x))

n = n.reset_index(drop=True)

#Función creada para cambiar los datos de [N,A,C,T,G]->[0,1,2,3,4]

def cambio1(dataT):
    for i in range(0,dataT.shape[0]):
        dataT['secuencia'][i] = dataT['secuencia'][i].replace('N','0')
        dataT['secuencia'][i] = dataT['secuencia'][i].replace('A','1')
        dataT['secuencia'][i] = dataT['secuencia'][i].replace('C','2')
        dataT['secuencia'][i] = dataT['secuencia'][i].replace('T','3')
        dataT['secuencia'][i] = dataT['secuencia'][i].replace('G','4')
    dataT['secuencia'] = dataT['secuencia'].apply(lambda x: x.split(','))
    return dataT

resultado = cambio1(df)
resultado_n = cambio1(n)
resultado_n2 = resultado_n['secuencia'].apply(lambda x: pd.Series(x)) 

resultado2 = resultado['secuencia'].apply(lambda x: pd.Series(x)) 

resultado2.to_csv(carpeta_salida+'/0123_NONuc_Pombe2.csv', header = True, index = False)
resultado_n2.to_csv(carpeta_salida+'/0123_Nuc_Pombe2.csv', header = True, index = False)
#resultado2.to_csv('/lhome/ext/usal053/usal0531/TFG/data/0123_NONuc_Pombe2.csv', header = True, index = False)

# ## Esto hay que hacer para pasarlo luego al ML
# resultado2 = resultado['secuencia'].apply(lambda x: pd.Series(x))
# resultado2.head()

# ## Segunda Forma
# 
# A -> 1000
# 
# C -> 0100
# 
# G -> 0010
# 
# T -> 0001
# 
# N -> 0000


df = pd.read_csv(carpeta+'/PombeSeq3.csv')
#df = pd.read_csv('/lhome/ext/usal053/usal0531/TFG/data/PombeSeq3.csv')
df.rename(columns = {'0':'secuencia'}, inplace = True)
df1 = pd.read_csv(carpeta+'/NDRseq2.csv')
#df = pd.read_csv('/lhome/ext/usal053/usal0531/TFG/data/NDRseq2.csv')
df = pd.concat([df, df1], axis=0)

#Nucleosomas
n = pd.read_csv(carpeta+'/NucleocSeq.csv')


df['secuencia'] = df["secuencia"].apply(lambda x: re.sub(r'([A-Z])(?!$)', r'\1,', x))
df['secuencia'] = df['secuencia'].apply(lambda x: x.split(','))

n['secuencia'] = n["secuencia"].apply(lambda x: re.sub(r'([A-Z])(?!$)', r'\1,', x))
n['secuencia'] = n['secuencia'].apply(lambda x: x.split(','))

def cambio2(dataT):
    seq = pd.DataFrame(dataT['secuencia'].values.tolist(), index=dataT.index)
    seq = seq.apply(lambda x: x.replace('A','1000'))
    seq = seq.apply(lambda x: x.replace('C','0100'))
    seq = seq.apply(lambda x: x.replace('G','0010'))
    seq = seq.apply(lambda x: x.replace('T','0001'))
    seq = seq.apply(lambda x: x.replace('N','0000'))
    return seq

resulSeq = cambio2(df)
resulNuc = cambio2(n)


resulSeq.to_csv(carpeta_salida+'/binario_NONuc_Pombe2.csv', header = True, index = False)
#resulSeq('/lhome/ext/usal053/usal0531/TFG/data/binario_NONuc_Pombe2.csv')
resulNuc.to_csv(carpeta_salida+'/binario_Nuc_Pombe2.csv', header = True, index = False)

# ## Tercera forma
# 
# [A,T,G,C,N] que se convierta en [0.25,0.5,0.75,1.0,0.0]
# Hay que convertirlo en un numpy array

df = pd.read_csv(carpeta+'/PombeSeq3.csv')
#df = pd.read_csv('/lhome/ext/usal053/usal0531/TFG/data/PombeSeq3.csv')
df.rename(columns = {'0':'secuencia'}, inplace = True)
df1 = pd.read_csv(carpeta+'/NDRseq2.csv')
#df = pd.read_csv('/lhome/ext/usal053/usal0531/TFG/data/NDRseq2.csv')
df = pd.concat([df, df1], axis=0)
df = df.reset_index(drop=True)

#Nucleosomas
n = pd.read_csv(carpeta+'/NucleocSeq.csv')

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','g','t','z']))


def cambiar_encoding(cadena):
    integer_encoded = label_encoder.transform(cadena)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0] = 0.25
    float_encoded[float_encoded == 1] = 0.50
    float_encoded[float_encoded == 2] = 0.75
    float_encoded[float_encoded == 3] = 1.00
    float_encoded[float_encoded == 4] = 0.00
    float_encoded[float_encoded == 5] = 0.00
    return float_encoded


df2 = pd.DataFrame(df)


for i in range(df.shape[0]):
    df['secuencia'][i] = cambiar_encoding(string_to_array(df['secuencia'][i]))

for i in range(n.shape[0]):
    n['secuencia'][i] = cambiar_encoding(string_to_array(n['secuencia'][i]))

df2 = df['secuencia'].apply(lambda x: pd.Series(x))
dfN = n['secuencia'].apply(lambda x: pd.Series(x))

df2.to_csv(carpeta_salida+'/float_NONuc_Pombe2.csv', header = True, index = False)
#df2.to_csv('/lhome/ext/usal053/usal0531/TFG/data/float_NONuc_Pombe2.csv', header = True, index = False)
dfN.to_csv(carpeta_salida+'/float_Nuc_Pombe2.csv', header = True, index = False)

### Cuarta forma
## Método k-mers para 2 nucleótidos, es decir, se va hacer cadenas de las secuencias con 149 parejas de nucleótidos

dataN = pd.read_csv(carpeta+'/NucleocSeq.csv')
data = pd.read_csv(carpeta+'/PombeSeq3.csv')
NDR = pd.read_csv(carpeta+'/NDRseq2.csv')

data.columns = ['secuencia']
data = pd.concat([NDR,data],axis=0)
dataN['nucleosoma'] = 1
data['nucleosoma'] = 0

###Cogemos 60000 filas del dataset de no nucleosomas
data = data.iloc[0:60000,:]
dataN = dataN.iloc[:,2:]
total = pd.concat([dataN,data],axis=0)

# function to convert sequence strings into k-mer words, el tamaño de las palabras se pone en el argumento size
def getKmers(sequence, size=2):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

total['dinuc'] = total.apply(lambda x:getKmers(x['secuencia']),axis=1)
total = total.drop('secuencia', axis=1)
total = total.sample(frac=1).reset_index(drop=True)

##En este caso ya está todo en el dataset los nucleosomas como los no-nucleosomas
total.to_csv(carpeta_salida+'/dinucTotal2.csv')


