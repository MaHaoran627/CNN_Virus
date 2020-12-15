import keras
import numpy as np

#dictionary for one-hot encoding
d_nucl={"A":0,"C":1,"G":2,"T":3,"N":4}

#get different learning weights for different classes
def get_learning_weights(filepath):
	f=open(filepath,"r").readlines()
	d_weights={}
	for i in f:
        	i=i.strip().split("\t")
        	d_weights[float(i[0])]=float(i[1])
	return d_weights

#set default params for generating batches of 50-mer
def get_params_50mer():
	params = {'batch_size': 1024,
	'n_classes': 187,
	'shuffle': True}
	return params

#set default params for generating batches of 100-mer
def get_params_150mer():
        params = {'batch_size': 101,
        'n_classes': 187,
        'shuffle': False}
        return params

#get k-mers, labels and locations for 50-mer
#default format for each line of training files: kmer+"\t"+label+"\t"+location
def get_kmer_from_50mer(filepath):
	f=open(filepath,"r").readlines()
	f_matrix=[]
	f_labels=[]
	f_pos=[]
	for i in f:
		i=i.strip().split("\t")
		f_matrix.append(i[0])
		f_labels.append(i[1])
		f_pos.append(i[2])
	return f_matrix,f_labels,f_pos

#get k-mers, labels and locations for 150-mer
#default format for each line of training files: kmer+"\t"+label+"\t"+location
def get_kmer_from_150mer(filepath):
	f=open(filepath,"r").readlines()
	f_matrix=[]
	f_labels=[]
	f_pos=[]
	for line in f:
		line=line.strip().split("\t")
		f_labels.append(line[1])
		f_pos.append(line[2])
		for i in range(len(line[0])-49):
			kmer=line[0][i:i+50]
			f_matrix.append(kmer)
	return f_matrix,f_labels,f_pos

#get k-mers from RNA-seq files of COV-ID-19 patients
#default format for each line of training files: kmer
def get_kmer_from_realdata(filepath):
	f=open(filepath,"r").readlines()
	lines=[]
	for i in range(0,len(f),4):
		lines.append(f[i+1].strip())
	f_matrix=[]
	f_index=[]
	sum_loc=0
	for line in lines:
		line=line.strip()
		length_of_read=len(line)
		if length_of_read>=50:
			for i in range(len(line)-49):
				kmer=line[i:i+50]
				f_matrix.append(kmer)
				sum_loc+=1
			f_index.append(sum_loc)
	return f_matrix,f_index

# simulated hiv-1 reads using santa-sim
# input: fastq format
def get_kmer_from_santi(filepath):
	f=open(filepath,"r").readlines()
	lines=[]
	for i in range(0,len(f),4):
		lines.append(f[i+1].strip())
	f_matrix=[]
	f_index=[]
	sum_loc=0
	for line in lines:
		line=line.strip()
		length_of_read=len(line)
		if length_of_read>=50:
			for i in range(len(line)-49):
				kmer=line[i:i+50]
				f_matrix.append(kmer)
				sum_loc+=1
			f_index.append(sum_loc)
	return f_matrix,f_index

#data generator for generating batches of data from 50-mers
class DataGenerator_from_50mer(keras.utils.Sequence):
	def __init__(self, f_matrix, f_labels, f_pos, batch_size=1024,n_classes=187, shuffle=True):
		self.batch_size = batch_size
		self.labels = f_labels
		self.matrix = f_matrix
		self.pos = f_pos
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()
	def __len__(self):
		return int(np.ceil(len(self.labels) / self.batch_size))
	def __getitem__(self, index):
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		X, y= self.__data_generation(indexes)
		return X,y
	def on_epoch_end(self):
		self.indexes = np.arange(len(self.labels))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
	def __data_generation(self, index):
		x_train=[]
		for i in index:
			seq=self.matrix[i]
			seq_list=[j for j in seq]
			x_train.append(seq_list)
		x_train=np.array(x_train)
		x_tensor=np.zeros(list(x_train.shape)+[5])
		for row in range(len(x_train)):
			for col in range(50):
				x_tensor[row,col,d_nucl[x_train[row,col]]]=1
		y_pos=[]
		y_label=[self.labels[i] for i in index]
		y_label=np.array(y_label)
		y_label=keras.utils.to_categorical(y_label, num_classes=self.n_classes)
		y_pos=[self.pos[i] for i in index]
		y_pos=np.array(y_pos)
		y_pos=keras.utils.to_categorical(y_pos, num_classes=10)
		return x_tensor,{'output1': y_label, 'output2': y_pos}

#data generator for generating batches of data from 50-mers for testing
class DataGenerator_from_50mer_testing(keras.utils.Sequence):
	def __init__(self, f_matrix, batch_size=1024,shuffle=False):
		self.batch_size = batch_size
		self.matrix = f_matrix
		self.shuffle = shuffle
		self.on_epoch_end()
	def __len__(self):
		return int(np.ceil(len(self.matrix) / self.batch_size))
	def __getitem__(self, index):
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		X = self.__data_generation(indexes)
		return X
	def on_epoch_end(self):
		self.indexes = np.arange(len(self.matrix))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
	def __data_generation(self, index):
		x_train=[]
		for i in index:
			seq=self.matrix[i]
			seq_list=[j for j in seq]
			x_train.append(seq_list)
		x_train=np.array(x_train)
		x_tensor=np.zeros(list(x_train.shape)+[5])
		for row in range(len(x_train)):
			for col in range(50):
				x_tensor[row,col,d_nucl[x_train[row,col]]]=1
		return x_tensor

#data generator for generating batches of data from 100-mers
class DataGenerator_from_150mer(keras.utils.Sequence):
	def __init__(self, f_matrix, batch_size=101,n_classes=187, shuffle=False):
		self.batch_size = batch_size
		self.matrix = f_matrix
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()
	def __len__(self):
		return int(np.ceil(len(self.matrix) / self.batch_size))
	def __getitem__(self, index):
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		X = self.__data_generation(indexes)
		return X
	def on_epoch_end(self):
		self.indexes = np.arange(len(self.matrix))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
	def __data_generation(self, index):
		x_train=[]
		for i in index:
			seq=self.matrix[i]
			seq_list=[j for j in seq]
			x_train.append(seq_list)
		x_train=np.array(x_train)
		x_tensor=np.zeros(list(x_train.shape)+[5])
		for row in range(len(x_train)):
			for col in range(50):
				x_tensor[row,col,d_nucl[x_train[row,col]]]=1
		return x_tensor

#data generator for generating batches of data from real-world data
class DataGenerator_from_realdata(keras.utils.Sequence):
	def __init__(self, f_matrix,index_list,batch_size=51,n_classes=187, shuffle=False):
		self.batch_size = batch_size
		self.matrix = f_matrix
		self.index_list=index_list
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()
	def __len__(self):
		return len(self.index_list)
	def __getitem__(self, index):
		if index==0:
			indexes = self.indexes[0:self.index_list[index]]
		else:
			indexes = self.indexes[self.index_list[index-1]:self.index_list[index]]
		X = self.__data_generation(indexes)
		return X
	def on_epoch_end(self):
		self.indexes = np.arange(len(self.matrix))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
	def __data_generation(self, indexes):
		x_train=[]
		for i in indexes:
			seq=self.matrix[i]
			seq_list=[j for j in seq]
			x_train.append(seq_list)
		x_train=np.array(x_train)
		x_tensor=np.zeros(list(x_train.shape)+[5])
		for row in range(len(x_train)):
			for col in range(50):
				x_tensor[row,col,d_nucl[x_train[row,col]]]=1
		return x_tensor
