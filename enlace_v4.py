from __future__ import print_function

import csv
import time
import detector_v4 as d 

import elasticsearch                                #Módulo de elasticsearch
from elasticsearch import Elasticsearch, helpers    #Añadidos para facilitar volcado de datos

import numpy as np # Biblioteca de funciones matematicas de alto nivel
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
import keras
import pandas
from keras.preprocessing import sequence
from keras.models import Sequential # necesario para poder generar la red neuronal
from keras.layers import Dense, Dropout, Activation, Lambda # Tipos de capa, hacen lo siguiente:
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten, LSTM
from keras.callbacks import CSVLogger # para guardar los datos en un excel
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.datasets import imdb # un dataset incluido en keras
from keras import backend as K # importas el backend (Tensorflow, Theano, etc)
import pandas as pd # pandas es una libreria extension de numpy usada para manipulacion y analisis de datos, para manipular tablas numericas y series temporales
from keras.utils.np_utils import to_categorical # sirve para convertir vectores de enteros a una matriz de clases binaria
import h5py # para almacenar un monton de datos numericos y dar facilidades de manipulacion para datos de Numpy
from sklearn.preprocessing import Normalizer # Para normalizar los datos
from sklearn.model_selection import train_test_split #para hacer la separacion entre datos de test y train
from sklearn.preprocessing import OneHotEncoder #para convertir los datos de entrada

#Declaración de variables auxiliares
filePath='E:/TFG/CICFlowMeter/CICFlowMeter-master/build/distributions/CICFlowMeter-4.0/bin/data/daily/2022-04-19_Flow.csv'
# filePath='E:/TFG/CICFlowMeter/CICFlowMeter-master/build/install/CICFlowMeter/bin/data/daily/2022-04-02Flow.csv'
dataset_size=0
t_total=0
condicion=0

#Construcción red neuronal y la guardamos en model 
model=d.red_neuronal()
scaler_=d.normalizador()

#Abrimos el archivo y lo guardamos en una variable llamada f
with open(filePath) as f:
    while True:                                                 #Bucle que se ejecuta constantemente
        dataset=pd.read_csv(filePath, encoding = "ISO-8859-1")  #pd.read_csv lee el archivo csv entero 
        filas_n=dataset.shape[0]-dataset_size
        if dataset.shape[0]>dataset_size:                       #Si el número de filas del archivo leido (nº flujos) es mayor que el nº filas anterior entonces clasifico sino no 
            t_inicio=time.time()                                #Hallamos el tiempo antes de la clasificación
            if dataset_size==0:
                filas_n=filas_n+1
            else:
                condicion=1
            d.clasificador(dataset.tail(filas_n),model,dataset_size,condicion,scaler_)         #Línea que se encarga de llamar a la red neuronal para clasificar nuevos flujos
            t_fin=time.time()                                   #Hallamos el tiempo después de la clasificación
            t_total=(t_fin-t_inicio)+t_total;                     #Hallamos el tiempo total
            t_medio=(t_fin-t_inicio)/(dataset.shape[0]-dataset_size) #Obtenemos el tiempo medio, restamos el número de filas total del dataset menos el anterior porque son los nuevos flujos que clasifica, inicialmente el anterior vale 1 (fila de características)
            
            n_bytes=dataset['Subflow Fwd Byts']+dataset['Subflow Bwd Byts']     #Sumamos la columna de enviados y recibidos para tener el columna con el total de bytes
            n_bytes_tot=n_bytes.sum()                                           #Obtenemos la suma de todas las filas para hallar el total de bytes 
            
            print("Se han clasificado",(dataset.shape[0]-dataset_size),"flujos en cada uno",t_medio) #Obtenemos el número de flujos clasificados por tiempo
            print("El tiempo total ha sido ",t_total, "segundos")                           #Vemos el tiempo total
            print("El ancho de banda en bytes/s es ",n_bytes_tot/t_total)                   #Obtenemos el ancho de banda dividiendo el número de bytes total entre el tiempo total
            print("El ancho de banda en bits/s es ",(n_bytes_tot*8)/t_total)                #Obtenemos el ancho de banda en bits/s
            dataset_size=dataset.shape[0]                                                   #Actualizamos el valor del número de filas 

#El bucle se ejecuta constantemente, se lee el archivo csv que genera el CICFLOWMETER, si el número de filas del dataset leído es mayor que 1
#entonces ya tiene un flujo (la primera fila indica las características). Y se lee constantemente el dataset viendo el núemero de filas. Si 
#el número de filas es mayor que el almacenado en dataset_size (lectura anterior) es que han llegado flujos nuevos y entonces clasificamos.

#Para calcular el tiempo medio se saca el tiempo antes y después de la clasificación y se hace una resta. Para calcularlo cada nueva llegada 
#de flujos, lo que hacemos es restar el número total de flujos menos el que había en la lectura anterior, obteniendo los nuevos flujos que 
#se van a clasificar.
    


  



