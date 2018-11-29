import numpy as np
class DataSet:
	def __init__(self,path,dataset,loss_type='square_loss'):
		self.path = path+dataset+'/'
		self.trainfile = self.path + dataset+'.train.libfm'
		self.testfile = self.path + dataset+'.test.libfm'
		self.validationfile = self.path + dataset+'.validation.libfm'
		self.loss_type = loss_type
		self.feat_dim = self.map_features()
		self.train_data,self.validation_data,self.test_data = self.construct_data(loss_type)
	def map_features(self):
		self.features = {}
		self.read_features(self.trainfile)
		self.read_features(self.testfile)
		self.read_features(self.validationfile)
		return len(self.features)
	def read_features(self,file):
		i = len(self.features)
		with open(file,'r') as f:
			for line in f:
				items = line.split(' ')
				for item in items[1:]:
					if item not in self.features:
						self.features[item] = i
						i = i + 1
	def construct_data(self,loss_type):
		train_x,train_y,train_y_for_logloss = self.read_data(self.trainfile)
		valid_x,valid_y,valid_y_for_logloss = self.read_data(self.validationfile)
		test_x,test_y,test_y_for_logloss = self.read_data(self.testfile)
		if self.loss_type == 'log_loss':
			train_data = self.construct_dataset(train_x,train_y_for_logloss)
			validation_data = self.construct_dataset(valid_x,valid_y_for_logloss)
			test_data = self.construct_dataset(test_x,test_y_for_logloss)
		else:
			train_data = self.construct_dataset(train_x,train_y)
			validation_data = self.construct_dataset(valid_x,valid_y)
			test_data = self.construct_dataset(test_x,test_y)
		return train_data,validation_data,test_data

	def read_data(self,file):
		x = []
		y = []
		y_for_logloss = []
		with open(file,'r') as f:
			for line in f:
				items = line.split(' ')
				y.append(1.0*float(items[0]))
				if float(items[0]) > 0:
					v = 1.0
				else:
					v = 0.0
				y_for_logloss.append(v)
				x.append([self.features[item] for item in items[1:]])
		return x,y,y_for_logloss
	def construct_dataset(self,x,y):
		data = {}
		x_lens = [len(line) for line in x]
		idxs = np.argsort(x_lens)
		data['x'] = [x[i] for i in idxs]
		data['y'] = [y[i] for i in idxs]
		return data
	def truncated_data(self):
		num_features = len(self.train_data['x'][0])
		for i in range(len(self.train_data['x'])):
			num_features = min(num_features,len(self.train_data['x'][i]))
		for i in range(len(self.train_data['x'])):
			self.train_data['x'][i] = self.train_data['x'][i][0:num_features]
		for i in range(len(self.validation_data['x'])):
			self.validation_data['x'][i] = self.validation_data['x'][i][0:num_features]
		for i in range(len(self.test_data['x'])):
			self.test_data['x'][i] = self.test_data['x'][i][0:num_features]
		return num_features	


