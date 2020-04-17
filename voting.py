import collections
def get_final_result(labels,probs1,loc,probs2):
	valid_labels=[]
	valid_loc=[]
	for i in range(len(probs1)):
		if float(probs1[i])>=0.9:
			valid_labels.append(labels[i])
			valid_loc.append(loc[i])
	if len(valid_labels)==0:
		return "187","10"
	else:
		d_count={}
		for i in range(len(valid_labels)):
			if valid_labels[i] in d_count:
				d_count[valid_labels[i]].append(valid_loc[i])
			else:
				d_count[valid_labels[i]]=[]
				d_count[valid_labels[i]].append(valid_loc[i])
		counter=collections.Counter(valid_labels)
		true_label=counter.most_common(1)[0][0]
		counter_loc=collections.Counter(d_count[true_label])
		true_loc=counter_loc.most_common(1)[0][0]
		return true_label,true_loc
