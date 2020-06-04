import keras
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Flatten, Dropout, Activation, BatchNormalization, Input
from keras import models
from keras.models import Model
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import numpy as np
import random
from preprocessing import get_kmer_from_50mer,get_params_50mer,get_learning_weights,DataGenerator_from_50mer
from architecture import build_model

#path for the training file
filepath_train="./data/50mer_training"
#path for the validating file
filepath_val="./data/50mer_validating"

#path for the learning weights file
filepath_weights="./data/weights_of_classes"

#paths for saving model and loss
filepath_loss="./data/Multi_task_model.loss"
filepath_model="./data/checkpoint/best_model.h5"

d_nucl={"A":0,"C":1,"G":2,"T":3,"N":4}
f_matrix,f_labels,f_pos=get_kmer_from_50mer(filepath_train)
f_matrix_val,f_labels_val,f_pos_val=get_kmer_from_50mer(filepath_val)

params = get_params_50mer()
d_weights=get_learning_weights(filepath_weights)

training_generator = DataGenerator_from_50mer(f_matrix, f_labels, f_pos, **params)
val_generator = DataGenerator_from_50mer(f_matrix_val, f_labels_val, f_pos_val, **params)

model=build_model()
print(model.summary())
model.compile(optimizer='adam',
	loss={'output1':'categorical_crossentropy','output2':'categorical_crossentropy'},
	metrics=['accuracy'])

CallBacks = [EarlyStopping(monitor='val_loss', patience=5),
                ModelCheckpoint(filepath=filepath_model, monitor='val_loss', save_best_only=True)]

result = model.fit_generator(training_generator,
	epochs=100,
	verbose=1,
	#use_multiprocessing=True,
	#workers=4,
	class_weight=d_weights,
	validation_data=val_generator,
	callbacks=CallBacks
	)

with open(filepath_loss,"wb") as f:
	f.write(str(result.history).encode())
