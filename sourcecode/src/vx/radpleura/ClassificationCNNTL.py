
import pandas as pd
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.layers import *

class ClassificationCNNTL:
    def __init__(self):        
        # https://www.kaggle.com/code/th3niko/transfer-learning-xception
        pathtestdata = "xxxxxxx"
        test_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
        )
        test_set = test_datagen.flow_from_directory(pathtestdata, target_size=(width,height))
        self.inv_map = {v: k for k, v in test_set.class_indices.items()}
        self.labels = [0 for i in range(len(self.inv_map)) ]
        for k, v in self.inv_map.items():
            self.labels[k] = v

        del test_datagen
        del test_set


        #load models
        self.model = {}

        #load model Xception
        self.model["Xception150x224"] = keras.models.load_model('Xception224x224.h5')
        self.model["MobileNet150x224"] = keras.models.load_model('MobileNet224x224.h5')
        self.model["ResNet150x224"] = keras.models.load_model('ResNet224x224.h5')

        #np.set_printoptions(suppress=True)

    def __del__(self):
        for m in self.model:
            del m
        del self.model

    def readTiles(self, pathquery, idrois):
        X = np.load(os.path.join(pathquery,"roids.npy"))()
        tiles = []
        for i in idrois:
            #input_arr = input_arr.resize((224,224))
            input_arr = X[i]
            tiles.append(np.array([input_arr/255.0]))
        del X
        return tiles

    def predict(self, pathquery, idmodelversion, idmodel, idrois):
        ypred = []
        if idmodel in self.model:
            X = self.readTiles(pathquery, idrois)
            for imgarr in X:
                preds = self.model[idmodel].predict(imgarr)
                ypred.append(preds.argmax())
                #self.classes_inv_map[preds.argmax()]

        return ypred, self.labels
