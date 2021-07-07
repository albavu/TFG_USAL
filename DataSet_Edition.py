#############################################################################
            # Autor: Alba Vallejo Urruchi
            # TFG Ingeniería Informática 2020-2021
            # Universidad de Salamanca
#############################################################################

#############################################################################
            ###### DATOS PARA METER A LOS DIFERENTES MODELOS ########
#############################################################################

import re
import pandas as pd
import numpy as np
import random
from random import random_permutation

                    ##### Funcion para LEER FASTA ##########
## Esta función se ha cogido del archivo bioio.py creado por Rodrigo Santamaría Vicente
## Nos devuelve un diccionario de Python

def readFasta(path=""):
    f=open(path)
    reader=f.readlines()
    f=open(path)
    ff={}
    k=re.sub("\(.*\)", "", re.sub(">", "", reader[0])).strip()
    i=0
    seq=""
    while i<len(reader):
        while(i<len(reader)):
            line=next(f)
            i+=1
            if(line.startswith(">")):
                ff[k]=seq
                seq=""
                k=re.sub("\(.*\)", "", re.sub(">", "", line)).strip()
                break
            else:
                seq+=line.replace("\n","").upper()
    if(len(seq)==0):
        seq+=line.replace("\n","").upper()
    ff[k]=seq
    return ff

            ########### 1º PASO SECUENCIAS NUCLEOSOMALES ##############
# El fichero contiene las secuencias de 150 bp que posicionan nucleosomas

seqfile = '/lhome/ext/usal053/usal0531/TFG/data/Nuc_SP3.fasta'
data = readFasta(seqfile)

## data es un diccionario por eso ahora lo vamos a pasar a DataFrame para poder trabajar con ello
df = pd.DataFrame([[key, data[key]] for key in data.keys()], columns=['chromosome','secuencia'])

#df.shape
#(49998, 2)

column = df['chromosome'].str.split(':', expand=True)
column.columns = ['chromosome','posInic']
df['chromosome'] = column['chromosome']
df.insert(1,"posInic",column['posInic'],True)

##Lo metemos a un archivo
#El dataframe tiene 49998 filas que corresponde a las secuencias que posicionan nucleosomas
#Además de las secuencias en crudo, también tiene datos de la posicionan inicial que comienza el nucleosoma y el cromosoma al que pertenece
df.to_csv('/lhome/ext/usal053/usal0531/TFG/data/NucleocSeq.csv', header = True, index = False)

                ############ 2º PASO SECUENCIAS NDRs ##############
#El fichero fasta contiene las secuencias de NDRs de S.Pombe que vamos a usar
#Estas secuencias sabemos que son las que no posicionan nucleosomas

#ndrfile = '/lhome/ext/usal053/usal0531/TFG/data/NDRs_total_Spombe.fasta'
ndrfile = '/Users/alba_vu/Desktop/curso20-21/TFG/data/NDRs_total_Spombe.fasta'
ndr = readFasta(ndrfile)
dfNDR = pd.DataFrame([[key, ndr[key]] for key in ndr.keys()], columns=['gen','secuencia'])

## 2 Formas para conseguir secuencias de NDR
# 1º 10000 secuencias a partir de secuencias de 150 de toda las secuencias de NDR unidas de todos los genes

seqNDR = dfNDR['secuencia']
seqNDR = ''.join(seqNDR)
nuevo= []
for i in range(0,10000):
    aleat = random.randrange((len(seqNDR))-150)
    nuevo.append(seqNDR[aleat:aleat+150])
dfSeqNDR = pd.DataFrame(data=nuevo)
dfSeqNDR.columns = ['secuencia']

# 2º forma 1468 secuencias de NDR que tienen más de 150 nucleótidos

seqNDR = dfNDR[dfNDR["secuencia"].apply(str).map(len)>=150]['secuencia']
seqNDR = seqNDR.to_frame()
seqNDR.columns = ['secuencia']

## Unimos ambos DataFrame en 1 y cortamos las secuencias para que sean de 150 pb
seqNDR2 = pd.concat([dfSeqNDR, seqNDR], axis=0)
seqNDR2['secuencia'] = seqNDR2['secuencia'].str.slice(0, 150)
#seqNDR2.to_csv('/lhome/ext/usal053/usal0531/TFG/data/NDRseq2.csv', header = True, index = False)
seqNDR2.to_csv('/Users/alba_vu/Desktop/curso20-21/TFG/data/NDRseq2.csv', header = True, index = False)

                ############ 3º PASO SECUENCIAS aleatorios de S.Pombe ##############
#Para ampliar un poco nuestro conjunto de datos, vamos a tener secuencias aleatorias extraídas de todo el genoma de S.Pombe
#De todo el genoma extraemos secuencias de longitud 150 bp

pombe = readFasta('/Users/alba_vu/Desktop/curso20-21/TFG/data/Spombe.fasta')
dfPombe = pd.DataFrame([[key, pombe[key]] for key in pombe.keys()], columns=['cromosoma','secuencia'])

## Juntamos las secuencias de los tres crommosomas

chr1= dfPombe['secuencia'][2]
chr2= dfPombe['secuencia'][1]
chr3 = dfPombe['secuencia'][0]
nuevo= []
for i in range(0,70000):
    aleat = random.randrange((len(chr1))-150)
    nuevo.append(''.join(random_permutation(chr1[aleat:aleat+150])))

for i in range(0,70000):
    aleat = random.randrange((len(chr2))-150)
    nuevo.append(''.join(random_permutation(chr2[aleat:aleat+150])))

for i in range(0,70000):
    aleat = random.randrange((len(chr3))-150)
    nuevo.append(''.join(random_permutation(chr3[aleat:aleat+150])))

df = pd.DataFrame(data=nuevo)

df.to_csv('/Users/alba_vu/Desktop/curso20-21/TFG/data/PombeSeq3.csv', header = True, index = False)



