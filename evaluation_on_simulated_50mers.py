import keras
from keras.models import load_model
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
from preprocessing import get_kmer_from_50mer,get_params_50mer,DataGenerator_from_50mer_testing

#path for testing file
filepath_train="./data/50mer_testing"
#path for trained model
filepath_model="./data/pretrained_model.h5"

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

acc_kappa=cohen_kappa_score(y_true, y_pred)
precision_macro=precision_score(y_true, y_pred, average='macro')
precision_micro=precision_score(y_true, y_pred, average='micro')
recall_macro=recall_score(y_true, y_pred, average='macro')
recall_micro=recall_score(y_true, y_pred, average='micro')
f1_macro=f1_score(y_true, y_pred, average='macro')
f1_micro=f1_score(y_true, y_pred, average='micro')

print("kappa is %f" % acc_kappa)
print("precision_macro is %f" % precision_macro)
print("precision_micro is %f" % precision_micro)
print("recall_macro is %f" % recall_macro)
print("recall_micro is %f" % recall_micro)
print("f1_macro is %f" % f1_macro)
print("f1_micro is %f" % f1_micro)
