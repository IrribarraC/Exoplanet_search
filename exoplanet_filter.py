#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:18:52 2017

@author: irribarrac
"""

import sklearn
import os
import scipy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import pandas
import pickle
plt.style.use('ggplot')
print('tensorflow: %s' % tensorflow.__version__)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
import keras.backend as K
from keras.optimizers import SGD
from keras.preprocessing import sequence
from sklearn.model_selection import  StratifiedKFold
from sklearn.model_selection import cross_val_score
import h5py
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter
from matplotlib import rcParams
import matplotlib.cm
seed = 7
np.random.seed(seed)
os.chdir("/home/irribarrac/UC/Experimental/k2_data")
curdir= os.getcwd()
traindir= os.path.abspath("training")
codes = ["eclipsing_binaries", "exoplanets"]
curves = []
curves_id = []
classes = []
os.chdir(traindir)
"""
for i in codes:
        index = codes.index(i)
        for filename in os.listdir(i):
            with open(os.path.join(i, filename), "rb") as datafile:
                lightcurve = pickle.load(datafile, encoding = 'latin1')
            filt = gaussian_filter(medfilt(lightcurve['fluxes'],17),5)
            flux = 1000.0* (lightcurve['fluxes']/filt -1)     
            curves_id.append(filename)
            curves.append(flux)
            classes.append(index)
            
X= sequence.pad_sequences(curves, maxlen=3424, dtype = "float64", value = 1) 
Y = np.asarray(classes)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
"""
bsize = 18
num_train= 1494
#%%

#This part is for k-fold cross validation
K.clear_session()
model = None
model = Sequential()
model.add(Conv1D(96, 27, input_shape=(3424,1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(96, 9))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(96, 27))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.6))                #Capas de dropout son para evitar overfitting
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=3e-3, decay=1e-7 , momentum=5e-1, nesterov=False)  # Descenso de gradiente para optimizar
model.load_weights("weights_v2.h5")
model.compile(loss = "binary_crossentropy", optimizer = sgd, metrics = ['accuracy'])
model.summary()
#history= model.fit(x = np.expand_dims(X,axis= 2), y = Y, epochs =30, batch_size=bsize,  validation_split=0.1, shuffle = True)
#model.save_weights("weights_v2.h5")

Test_curves_id = []
Test_curves = []
os.chdir(curdir)
with open("EPICIDS.txt", "r") as EPICIDS:
    for line in EPICIDS:
        fname = line.replace("\n", "")
        Test_curves_id.append(fname)
        fname +=".pkl"
        with open (fname, "rb") as curve:
            lightcurve = pickle.load(curve, encoding = 'latin1')
        filt = gaussian_filter(medfilt(lightcurve['fluxes'],17),5)
        flux = 1000.0* (lightcurve['fluxes']/filt -1)      
        Test_curves.append(flux)
X_test = sequence.pad_sequences(Test_curves, maxlen=3424, dtype = "float64", value = 1)

referees = []
with open("Labeled.txt",'r') as knowns:
    for line in knowns:
        a = line.split()
        epicid = a[0]
        i = 0
        while epicid != Test_curves_id[i]:
            i+=1
        else:
            referees.append([i,a[1]])
plt.figure()
for i in range(2):
    Predictions = model.predict(x = np.expand_dims(X_test, axis = 2))
    p = Predictions.tolist()
    predictions = []
    for i in p:
        predictions.append(i[0])
    for n in range(len(referees)):
        print(Test_curves_id[int(referees[n][0])]," & ", referees[n][1]," & " ,'{:.5f}'.format(float(predictions[int(referees[n][0])])),"\\\\")
        plt.hist(predictions, bins = 20, range = (0,1), normed = False, cumulative = False, rwidth = 0.8)
plt.axvline(x=0.5,color = 'k')
plt.title("Prediction densities for 5 cases")
plt.xlabel("Prediction")
plt.ylabel("Fraction of the dataset")
plt.show()






"""
K.clear_session()
model = None
model = Sequential()
model.add(Conv1D(96, 27, input_shape=(3424,1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(96, 9))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(96, 27))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.6))                #Capas de dropout son para evitar overfitting
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=3e-4, decay=1e-6 , momentum=5e-1, nesterov=True)  # Descenso de gradiente para optimizar

#model.load_weights('weights.h5')

model.compile(loss = "binary_crossentropy", optimizer = sgd, metrics = ['binary_accuracy'])
model.summary()
model.fit(x = np.expand_dims(X,axis= 2), y = Y, epochs =30, batch_size=bsize)

model._make_predict_function()
#model.save_weights('weights.h5')


Test_curves_id = []
Test_curves = []
os.chdir(curdir)
with open("EPICIDS.txt", "r") as EPICIDS:
    for line in EPICIDS:
        fname = line.replace("\n", "")
        fname +=".pkl"
        with open (fname, "rb") as curve:
            lightcurve = pickle.load(curve, encoding = 'latin1')
        filt = gaussian_filter(medfilt(lightcurve['fluxes'],17),5)
        flux = 1000.0* (lightcurve['fluxes']/filt -1)      
        Test_curves_id.append(fname)
        Test_curves.append(flux)
X_test = sequence.pad_sequences(Test_curves, maxlen=3424, dtype = "float64", value = 1)
Predictions = model.predict(x = np.expand_dims(X_test, axis = 2))



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""