import os
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from google.cloud import storage
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error

class RecommenderModel:
    def __init__(self,modelConfig):
        self.modelConfig = modelConfig
        self.classification_threshold=0.5
        
    def buildNetwork(self):
        networkList=list()
        for nw in range(self.modelConfig.no_of_networks):
            network = Sequential()
            for ly in range(len(self.modelConfig.layers[nw])):
                dense=Dense(self.modelConfig.neurons[nw][ly],self.modelConfig.layers[nw][ly],kernel_regularizer=regularizers.l2(self.modelConfig.regularizers[nw][ly]))
                network.add(dense)
            networkList.append(network)
        return networkList
        
    def train(self,x_dataList,y_data):
        networkList = self.buildNetwork()
        nwInputList=list()
        nwOutputList=list()
            
        for index in range(self.modelConfig.no_of_networks):
            m,n = x_dataList[index].shape
            nwInput = Input(shape=(n))
            nwInputList.append(nwInput)
            nwOutput = networkList[index](nwInput)
            nwOutput = tf.linalg.l2_normalize(nwOutput, axis=1)
            nwOutputList.append(nwOutput)
        
        output=None
        if(self.modelConfig.no_of_networks == 2):
            output=tf.keras.layers.Dot(axes=1)(nwOutputList)
        else:
            output = nwOutputList
        self.model = tf.keras.Model(nwInputList, output)
        self.model.summary()
        self.model.compile(
            loss=self.modelConfig.loss,
            optimizer=self.modelConfig.optimizer
        )
        tf.random.set_seed(1)
        self.model.fit(x_dataList, y_data, epochs=self.modelConfig.epochs)         
        return self.getErr(x_dataList,y_data)

    def save(self):
        output_directory = os.environ['AIP_MODEL_DIR']
        self.model.save(output_directory)

    def validate(self,x_dataList,y_data):
        x_dataList_n=list()
        for x_data in x_dataList:
            x_dataList_n.append(self.normalize(x_data))
        return self.getErr(x_dataList_n,y_data)

    def test(self,x_dataList,y_data):
        x_dataList_n=list()
        for x_data in x_dataList:
            x_dataList_n.append(self.normalize(x_data))
        return self.getErr(x_dataList_n,y_data)

    def load(self):
        model_directory = os.environ['AIP_MODEL_DIR']
        self.model = tf.keras.models.load_model(model_directory)

    def normalize(self,x_data):
        norm_l = tf.keras.layers.Normalization(axis=-1)
        norm_l.adapt(x_data)  # learns mean, variance
        return norm_l(x_data)

    def getErr(self,x_dataList_n,y_data):
        if (self.modelConfig.model_type=="regression"):
            yhat = self.model.predict(x_dataList_n)
            return mean_squared_error(y_data, yhat) / 2
        if (self.modelConfig.model_type=="classification"):
            yhat = self.model.predict(x_dataList_n)
            yhat = np.where(yhat >= self.classification_threshold, 1, 0)
            return np.mean(yhat != y_data)
        if (self.modelConfig.model_type=="multi-classification"):
            model_predict_lambda = lambda Xl: np.argmax(tf.nn.softmax(self.model.predict(Xl)).numpy(),axis=1)
            yhat=model_predict_lambda(x_dataList_n)
            m = len(y_data)
            incorrect = 0
            for i in range(m):
                if yhat[i] != y_data[i]:
                    incorrect += 1
            cerr=incorrect/m
            return(cerr)
