import keras
from keras.models import load_model
import numpy as np
from preprocessing import get_kmer_from_50mer,get_params_50mer,DataGenerator_from_50mer_testing
from generate_report import save_report

#path for testing file
filepath_train="./data/50mer_testing"
#path for trained model
filepath_model="./data/pretrained_model.h5"
#path for report
filepath_report="./data/50mer_report"

d_nucl={"A":0,"C":1,"G":2,"T":3,"N":4}
f_matrix,y_true,y_true_loc=get_kmer_from_50mer(filepath_train)
#params=get_params_50mer()
testing_generator = DataGenerator_from_50mer_testing(f_matrix)

model=load_model(filepath_model)
hist = model.predict_generator(testing_generator,
	verbose=1
	)
y_pred=[str(i.argmax(axis=-1)) for i in hist[0]]
y_pred_loc=[str(i.argmax(axis=-1)) for i in hist[1]]

save_report(filepath_report,y_pred,y_pred_loc)
