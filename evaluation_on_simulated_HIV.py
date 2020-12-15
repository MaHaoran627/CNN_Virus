import keras
from keras import models
from keras.models import load_model
import random
from preprocessing import get_kmer_from_santi,DataGenerator_from_realdata,get_kmer_from_realdata
from voting import get_final_result
from generate_report import save_report

#path for testing file
filepath_train="./data/HIV-1-Dataset/hiv1_150mer_100_generations.fastq"

#path for trained model
filepath_model="./data/pretrained_model.h5"
#path for saving final report
filepath_report="./data/hiv-150mer-100g.report"

d_nucl={"A":0,"C":1,"G":2,"T":3,"N":4}
f_matrix,f_index=get_kmer_from_santi(filepath_train)
testing_generator = DataGenerator_from_realdata(f_matrix,f_index)

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
num_iters=len(f_index)
for i in range(0,num_iters):
	if i==0:
		tmp_label,tmp_loc=get_final_result(predicted_labels_list[0:f_index[i]],predicted_prob_list[0:f_index[i]],predicted_loc_list[0:f_index[i]],predicted_loc_prob_list[0:f_index[i]])
	else:
		tmp_label,tmp_loc=get_final_result(predicted_labels_list[f_index[i-1]:f_index[i]],predicted_prob_list[f_index[i-1]:f_index[i]],predicted_loc_list[f_index[i-1]:f_index[i]],predicted_loc_prob_list[f_index[i-1]:f_index[i]])
	final_label.append(tmp_label)
	final_loc.append(tmp_loc)

save_report(filepath_report,final_label,final_loc)
