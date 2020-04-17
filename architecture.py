import keras
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Flatten, Dropout, Activation, BatchNormalization, Input
from keras import models
from keras.models import Model
from keras.layers.pooling import MaxPooling1D
from keras.models import load_model
import numpy as np
import random

def build_model():
	print("begin to train")
	#build cnn model
	input_seq=Input(shape=(50,5))
	layer1=Convolution1D(512, 5, padding="same",activation="relu",kernel_initializer="he_uniform")(input_seq)
	layer2=BatchNormalization(momentum=0.6)(layer1)
	layer3=MaxPooling1D(pool_size=2,padding='same')(layer2)
	layer4=Convolution1D(512, 5, padding="same",activation="relu",kernel_initializer="he_uniform")(layer3)
	layer5=BatchNormalization(momentum=0.6)(layer4)
	layer6=MaxPooling1D(pool_size=2,padding='same')(layer5)
	layer7=Convolution1D(1024, 7, padding="same",activation="relu",kernel_initializer="he_uniform")(layer6)
	layer8=Convolution1D(1024, 7, padding="same",activation="relu",kernel_initializer="he_uniform")(layer7)
	layer9=BatchNormalization(momentum=0.6)(layer8)
	layer10=MaxPooling1D(pool_size=2,padding='same')(layer9)
	layer11=Flatten()(layer10)
	layer12=Dense(1024,kernel_initializer="he_uniform")(layer11)
	layer13=BatchNormalization(momentum=0.6)(layer12)
	layer14=Dropout(0.2)(layer13)
	output1=Dense(187, activation='softmax',kernel_initializer="he_uniform",name="output1")(layer14)
	output_con=keras.layers.concatenate([layer14,output1],axis=1)
	layer15=Dense(1024, kernel_initializer="he_uniform")(output_con)
	layer16=BatchNormalization(momentum=0.6)(layer15)
	output2=Dense(10, activation='softmax',kernel_initializer="he_uniform",name="output2")(layer16)
	model = Model(inputs=input_seq, outputs=[output1,output2])
	return model
