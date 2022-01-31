## Movement in Multiple Time (MiMT) learning algorithm
## Copyright (c) 2019, 2020 University College London
## Author: Temitayo Olugbade
    
import tensorflow as tf
from tensorflow import keras



class MiMTnn:


    def __init__(self, _numofseq, _seqlen, _numclasses=2, _numofscales=2):

        self.name = 'mimt'
        self.numclasses = _numclasses

        self.network = self.create_network(_numofseq, _seqlen, numofouts=_numofscales)


    def create_network(self, numofseq, seqlen, numofouts=1):
        
        ilayer = keras.Input(shape=(seqlen, 6, 3))
        gnlayer = keras.layers.GaussianNoise(0.01)(ilayer)
        reshapelayer1 = keras.layers.Reshape((seqlen, -1))(gnlayer)

        
        timeenclayer1 = keras.layers.LSTM(3, use_bias=True, unit_forget_bias=True, return_sequences=True, stateful=False)(reshapelayer1)
        timeenclayer2 = keras.layers.LSTM(3, use_bias=True, unit_forget_bias=True, return_sequences=True, stateful=False)(timeenclayer1)
        timeenclayer3 = keras.layers.LSTM(3, use_bias=True, unit_forget_bias=True, return_sequences=True, stateful=False)(timeenclayer2)
              
        timeencoder = keras.Model(ilayer, timeenclayer3)
                
        headilayer = keras.Input(shape=(seqlen, 6, 3))
        headoute = timeencoder(headilayer)
        lupperilayer = keras.Input(shape=(seqlen, 6, 3))
        lupperoute = timeencoder(lupperilayer)
        rupperilayer = keras.Input(shape=(seqlen, 6, 3))
        rupperoute = timeencoder(rupperilayer)
        llowerilayer = keras.Input(shape=(seqlen, 6, 3))
        lloweroute = timeencoder(llowerilayer)
        rlowerilayer = keras.Input(shape=(seqlen, 6, 3))
        rloweroute = timeencoder(rlowerilayer)


        mergelayer = keras.layers.concatenate([headoute, lupperoute, rupperoute, lloweroute, rloweroute], axis=-1)
        

        mergelayer2a = keras.layers.dot([mergelayer, mergelayer], axes=1)
        scalelayera = keras.layers.BatchNormalization(trainable=False)(mergelayer2a)
        softmaxlayera = keras.layers.Lambda(lambda x: keras.activations.softmax(x, axis=-1))(scalelayera)
        mergelayer3a = keras.layers.dot([mergelayer, softmaxlayera], axes=2)
        
        mergelayer2b = keras.layers.dot([mergelayer, mergelayer], axes=1)
        scalelayerb = keras.layers.BatchNormalization(trainable=False)(mergelayer2b)
        softmaxlayerb = keras.layers.Lambda(lambda x: keras.activations.softmax(x, axis=-1))(scalelayerb)
        mergelayer3b = keras.layers.dot([mergelayer, softmaxlayerb], axes=2)

        mergelayer2c = keras.layers.dot([mergelayer, mergelayer], axes=1)
        scalelayerc = keras.layers.BatchNormalization(trainable=False)(mergelayer2c)
        softmaxlayerc = keras.layers.Lambda(lambda x: keras.activations.softmax(x, axis=-1))(scalelayerc)
        mergelayer3c = keras.layers.dot([mergelayer, softmaxlayerc], axes=2)


        mergelayer4 = keras.layers.maximum([mergelayer3a, mergelayer3b, mergelayer3c])
        
        
        labeldeclayer = keras.layers.LSTM(15, use_bias=True, unit_forget_bias=True, return_sequences=True, stateful=False)(mergelayer4)        
        labelcontlayer1 = keras.layers.TimeDistributed(keras.layers.Dense(15))(labeldeclayer)
        labelcontlayer2 = keras.layers.GlobalAveragePooling1D(data_format='channels_first')(labelcontlayer1)
        labelcontout = keras.layers.Activation('sigmoid', name='label_continuous')(labelcontlayer2)
     
                        
        mergelayer5 = keras.layers.multiply([mergelayer, labelcontlayer1])
        labelsinglayer1 = keras.layers.LSTM(self.numclasses, use_bias=True, unit_forget_bias=True, return_sequences=False, stateful=False)(mergelayer5)
        labelsingout = keras.layers.Lambda(lambda x: keras.activations.softmax(x, axis=-1), name='label_singular')(labelsinglayer1)

                              
        net = keras.Model(inputs=[headilayer, lupperilayer, rupperilayer, llowerilayer, rlowerilayer],
                            outputs=[labelcontout, labelsingout])

        return net



