#############################################################################
            # Autor: Alba Vallejo Urruchi
            # TFG Ingeniería Informática 2020-2021
            # Universidad de Salamanca
#############################################################################

#!/usr/bin/env python
# coding: utf-8

#### Preparación de los datos teniendo en cuenta que son dinucleótidos


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


dataN = pd.read_csv('/lhome/ext/usal053/usal0531/TFG/data/NucleocSeq.csv')
data = pd.read_csv('/lhome/ext/usal053/usal0531/TFG/data/PombeSeq3.csv')
NDR = pd.read_csv('/lhome/ext/usal053/usal0531/TFG/data/NDRseq2.csv')


data.columns = ['secuencia']
data = pd.concat([NDR,data],axis=0)
dataN['nucleosoma'] = 1
data['nucleosoma'] = 0
data = data.iloc[0:60000,:]
dataN = dataN.iloc[:,2:]
total = pd.concat([dataN,data],axis=0)

## Función para convertir todas las secuencias en trozos con 2 nucleótidos
def getKmers(sequence, size=2):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

total['dinuc'] = total.apply(lambda x:getKmers(x['secuencia']),axis=1)
total = total.drop('secuencia', axis=1)
total = total.sample(frac=1).reset_index(drop=True)

### Primero guardamos el csv porque los target estan en este csv, por tanto, cuando lo cojamos para 
## los modelos de Deep Learning habrá que abrir los dos archivos
total.to_csv('/lhome/ext/usal053/usal0531/TFG/data/dinucTotal2.csv')


# Función para convertir la secuencias de DNA del dataset en numpy array
# lo convierte a minúsculas, cambia cualquier caracter que no sea 'acgt' a 'n'
import re
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgt]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

# create a label encoder with 'acgtn' alphabet
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['aa','ag','ac','at','tt','ta','tc','tg','gg','gt','ga','gc','cc','ca','cg','ct','nn']))


# function to one-hot encode a DNA sequence string
# non 'acgt' bases (n) are 0000
# returns a L x 17 numpy array

def one_hot_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int,categories=[range(18)])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded = np.delete(onehot_encoded, -1, 1)
    return onehot_encoded


# ## Reemplazo de dinucleotidos por z
# Todos los dinucleótidos que contengan la n los sustituimos por z

for a in range(total.shape[0]):
    for i in range(len(total.iloc[1,1])):
        if(total.iloc[a,1][i]=="na"):
            total.iloc[a,1][i] = "nn"
        elif(total.iloc[a,1][i]=="nc"):
            total.iloc[a,1][i] = "nn"
        elif(total.iloc[a,1][i]=="ng"):
            total.iloc[a,1][i] = "nn"
        elif(total.iloc[a,1][i]=="nt"):
            total.iloc[a,1][i] = "nn"
        elif(total.iloc[a,1][i]=="an"):
            total.iloc[a,1][i] = "nn"
        elif(total.iloc[a,1][i]=="cn"):
            total.iloc[a,1][i] = "nn"
        elif(total.iloc[a,1][i]=="gn"):
            total.iloc[a,1][i] = "nn"
        elif(total.iloc[a,1][i]=="tn"):
            total.iloc[a,1][i] = "nn"
    



nueva2 = []
for i in range(total.shape[0]):
    nueva2.append(one_hot_encoder(total.iloc[i,1]))

#### Se guarda el objeto con la librería pickle_

import pickle
with open("dinuc2.pickle","wb") as f:
    pickle.dump(nueva2,f)


