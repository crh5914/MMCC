#!/bin/python3
from sklearn.linear_model import LinearRegression
import numpy as np
from datasets import DataSet
from sklearn.metrics import mean_squared_error
def to_matrix(data,dim):
	full_train_data = []
	num_examples = len(data['x'])
	for i in range(num_examples):
		row = np.zeros(dim)
		row[train_data['x'][i]] = 1
		full_train_data.append(row)
	return np.array(full_train_data),data['y']

ds = DataSet('./data/','frappe')
train_data = ds.train_data
num_train = len(train_data['y'])
validation_data = ds.validation_data
num_valid = len(validation_data['y'])
test_data = ds.test_data
num_test = len(test_data['y'])
dim = ds.feat_dim
train_x,train_y = to_matrix(train_data,dim)
valid_x,valid_y = to_matrix(validation_data,dim)
test_x,test_y = to_matrix(test_data,dim)
clf = LinearRegression()
clf.fit(train_data,train_y)
train_y_ = clf.predict(train_x)
valid_y_ = clf.predict(valid_x)
test_y_ = clf.predict(test_y_)
train_rmse = mean_squared_error(train_y,train_y_)
valid_rmse = mean_squared_error(valid_y,valid_y_)
test_rmse = mean_squared_error(test_y,test_y_)
print('train rmse:',train_rmse,'validation rmse:',valid_rmse,'test rmse:',test_rmse)