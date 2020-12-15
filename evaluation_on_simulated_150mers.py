import keras
from keras import models
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
import random
import collections
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from preprocessing import get_kmer_from_150mer,get_params_150mer,DataGenerator_from_150mer
from voting import get_final_result

#path for testing file
filepath_train="./data/ICTV_150mer_benchmarking"
#path for trained model
filepath_model="./data/pretrained_model.h5"

d_nucl={"A":0,"C":1,"G":2,"T":3,"N":4}
f_matrix,y_true,f_pos=get_kmer_from_150mer(filepath_train)
params=get_params_150mer()
testing_generator = DataGenerator_from_150mer(f_matrix, **params)

model=load_model(filepath_model)
hist = model.predict_generator(testing_generator,
	verbose=1
	)

predicted_labels_list=[i.argmax(axis=-1) for i in hist[0]]
predicted_prob_list=[max(i) for i in hist[0]]
predicted_loc_list=[i.argmax(axis=-1) for i in hist[1]]
predicted_loc_prob_list=[max(i) for i in hist[1]]

final_label=[]
final_loc=[]
num_iters=int(len(predicted_labels_list)*1.0/101)
for i in range(0,num_iters):
	tmp_label,tmp_loc=get_final_result(predicted_labels_list[i*101:(i+1)*101],predicted_prob_list[i*101:(i+1)*101],predicted_loc_list[i*101:(i+1)*101],predicted_loc_prob_list[i*101:(i+1)*101])
	final_label.append(str(tmp_label))
	final_loc.append(str(tmp_loc))

y_pred=final_label
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
