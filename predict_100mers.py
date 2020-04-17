import keras
from keras import models
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
import random
from generate_report import save_report
from preprocessing import get_kmer_from_100mer,get_params_100mer,DataGenerator_from_100mer
from voting import get_final_result

#path for testing file
filepath_train="./data/100mer_testing"
#path for trained model
filepath_model="./data/pretrained_model.h5"
#path for report
filepath_report="./data/100mer_report"

BATCH_SIZE=51
d_nucl={"A":0,"C":1,"G":2,"T":3,"N":4}
f_matrix,y_true,f_pos=get_kmer_from_100mer(filepath_train)
params=get_params_100mer()
testing_generator = DataGenerator_from_100mer(f_matrix, **params)

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
num_iters=int(len(predicted_labels_list)*1.0/51)
for i in range(0,num_iters):
	tmp_label,tmp_loc=get_final_result(predicted_labels_list[i*51:(i+1)*51],predicted_prob_list[i*51:(i+1)*51],predicted_loc_list[i*51:(i+1)*51],predicted_loc_prob_list[i*51:(i+1)*51])
	final_label.append(str(tmp_label))
	final_loc.append(str(tmp_loc))

save_report(filepath_report,final_label,final_loc)
