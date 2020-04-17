def save_report(filepath_report,final_label,final_loc):
	d_tax={}
	for i in range(0,len(final_label)):
		if final_label[i] in d_tax:
			if final_loc[i] in d_tax[final_label[i]]:
				d_tax[final_label[i]][final_loc[i]]+=1
			else:
				d_tax[final_label[i]][final_loc[i]]=1
		elif final_label[i]!="187":
			d_tax[final_label[i]]={}
			d_tax[final_label[i]][final_loc[i]]=1

	with open(filepath_report,"wb") as f:
		for label in d_tax:
			tmp1=str(label)
			tmp2=""
			sum=0
			for loc in d_tax[label]:
				sum+=d_tax[label][loc]
				tmp2+="\t"+str(loc)+"\t"+str(d_tax[label][loc])
			tmp=tmp1+"\t"+str(sum)+tmp2+"\n"
			f.write(tmp.encode())
