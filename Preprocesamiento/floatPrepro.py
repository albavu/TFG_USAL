#############################################################################
            # Autor: Alba Vallejo Urruchi
            # TFG Ingeniería Informática 2020-2021
            # Universidad de Salamanca
#############################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
import pandas as pd
import numpy as np

#Variables
carpeta='/Users/alba_vu/Desktop/curso20-21/TFG/data/input'
#carpeta_s = '/lhome/ext/usal053/usal0531/TFG/data/datosEntrada'
carpeta_salida = '/Users/alba_vu/Desktop/curso20-21/TFG/data/output'

def string_to_array(cadena):
    cadena = cadena.lower()
    cadena = re.sub('[^acgt]','z',cadena)
    cadena = np.array(list(cadena))
    return cadena

# ## Tercera forma
# 
# [A,T,G,C,N] que se convierta en [0.25,0.5,0.75,1.0]
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
