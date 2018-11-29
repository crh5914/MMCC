import tensorflow as tf
from datasets import DataSet
import numpy as np
from time import time
import os
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def parse_args():
	parser = argparse.ArgumentParser(description='runn mutilple channle cnn')
	# parser = argparse.ArgumentParser(description="Run MLP.")
	# parser.add_argument('--path', nargs='?', default='Data/',
	# 					help='Input data path.')
	parser.add_argument('--dataset', nargs='?', default='ml-tag',
						help='Choose a dataset.')
	parser.add_argument('--epochs', type=int, default=100,
						help='Number of epochs.')
	parser.add_argument('--embedding_dim', type=int, default=64,
						help='embedding dimension.')
	parser.add_argument('--batch_size', type=int, default=256,
						help='Batch size.')
	parser.add_argument('--hfilter_size', nargs='?', default='[2,3]',
						help="filter size for horizatal cnn")
	parser.add_argument('--hfilter_num', nargs='?', default='[32,32]',
						help="filter number for horizatal cnn")
	parser.add_argument('--vfilter_size', nargs='?', default='[1,]',
						help="filter size for vertical cnn")
	parser.add_argument('--vfilter_num', nargs='?', default='[32,]',
						help="filter number for vertical cnn")
	parser.add_argument('--reg', type=float, default=0.01,
						help='reguralization parameter')
	parser.add_argument('--lr', type=float, default=0.001,
						help='learning rate.')
	parser.add_argument('--dropout', type=float, default=0.8,
						help='droupout rate')
	# parser.add_argument('--verbose', type=int, default=1,
	# 					help='Show performance per X iterations')
	# parser.add_argument('--out', type=int, default=1,
	# 					help='Whether to save the trained model.')
	return parser.parse_args()
class CCONN:
	def __init__(self,feat_dim,valid_dim,embedding_dim,hfilter_sizes,hfilter_nums,vfilter_sizes,vfilter_nums,epoch=1000,batch_size=256,lr=0.001,l2_reg_lambda=0.05,keep=0.8,verbose=1):
		self.feat_dim = feat_dim
		self.embedding_dim = embedding_dim
		self.hfilter_sizes = hfilter_sizes
		self.hfilter_nums = hfilter_nums
		self.vfilter_sizes = vfilter_sizes
		self.vfilter_nums = vfilter_nums
		self.epochs = epoch
		self.batch_size = batch_size
		self.lr = lr
		self.l2_reg_lambda = l2_reg_lambda
		self.keep = keep
		self.verbose = verbose
		self.pretrain_flag = False
		self.train_rmse,self.valid_rmse,self.test_rmse = [],[],[]
		self.valid_dim = valid_dim
		self._init_graph()
	def _init_graph(self):
		self.x = tf.placeholder(tf.int32,[None,self.valid_dim],name='x')
		self.y = tf.placeholder(tf.float32,[None,1],name='y')
		self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')
		l2_loss = tf.constant(0.0)
		count = 0
		with tf.name_scope('embedding'):
			W = tf.Variable(tf.random_uniform([self.feat_dim,self.embedding_dim],-1.0,1.0),name='W')
			self.embedded_feat = tf.nn.embedding_lookup(W,self.x)
		# 	element_wise_product_list = []
		# 	for i in range(0, self.valid_dim):
		# 		for j in range(i+1, self.valid_dim):
		# 			element_wise_product_list.append(tf.multiply(self.embedded_feat[:,i,:], self.embedded_feat[:,j,:]))
		# 			count += 1
		# 	element_wise_product = tf.stack(element_wise_product_list) 
		# 	element_wise_product = tf.transpose(element_wise_product, perm=[1,0,2], name="element_wise_product")
		# 	self.embedded_feat_expanded = tf.expand_dims(element_wise_product,-1)
		self.embedded_feat_expanded = tf.expand_dims(self.embedded_feat,-1)
		hpooled_outputs = []
		for filter_size,filter_num in zip(self.hfilter_sizes,self.hfilter_nums):
			with tf.name_scope('hconv-maxpool-%s'%filter_size):
				filter_shape = [filter_size,self.embedding_dim,1,filter_num]
				W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
				b = tf.Variable(tf.constant(0.1,shape=[filter_num]),name='b')
				conv = tf.nn.conv2d(self.embedded_feat_expanded,W,strides=[1,1,1,1],padding='VALID',name='conv')
				h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
				pooled = tf.nn.max_pool(h,ksize=[1,count-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')
				hpooled_outputs.append(pooled)
		hnum_filter_total = sum(self.hfilter_nums)
		self.h_pool = tf.concat(hpooled_outputs,3)
		self.h_pool_flat = tf.reshape(self.h_pool,[-1,hnum_filter_total])
		vpooled_outputs = []
		for filter_size,filter_num in zip(self.vfilter_sizes,self.vfilter_nums):
			with tf.name_scope('vconv-maxpool-%s'%filter_size):
				filter_shape = [count,filter_size,1,filter_num]
				W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
				b = tf.Variable(tf.constant(0.1,shape=[filter_num]),name='b')
				conv = tf.nn.conv2d(self.embedded_feat_expanded,W,strides=[1,1,1,1],padding='VALID',name='conv')
				h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
				# print(tf.shape(h))
				pooled = tf.nn.max_pool(h,ksize=[1,1,self.embedding_dim-filter_size+1,1],strides=[1,1,1,1],padding='VALID',name='pool')
				vpooled_outputs.append(pooled)
		vnum_filter_total = sum(self.vfilter_nums)
		self.v_pool = tf.concat(vpooled_outputs,3)
		self.v_pool_flat = tf.reshape(self.v_pool,[-1,vnum_filter_total])
		self.feat_flat = tf.concat([self.h_pool_flat,self.v_pool_flat],1)
		# self.feat_flat = self.h_pool_flat
		with tf.name_scope('dropout'):
			self.feat_flat_drop = tf.nn.dropout(self.feat_flat,self.dropout_keep_prob)
		num_feat_total = hnum_filter_total+vnum_filter_total
		# num_feat_total = hnum_filter_total
		# num_feat_total = self.embedding_dim
		# self.feat_flat = tf.reduce_sum(self.embedded_feat,axis=1)
		with tf.name_scope("output"):
			W1 = tf.get_variable('W1',shape=[num_feat_total,num_feat_total//2],initializer=tf.contrib.layers.xavier_initializer())
			b1 = tf.Variable(tf.constant(0.1,shape=[num_feat_total//2]),name='b1')
			W2 = tf.get_variable('W2',shape=[num_feat_total//2,1],initializer=tf.contrib.layers.xavier_initializer())
			b2 = tf.Variable(tf.constant(0.1,shape=[1]),name='b2')
			l2_loss += tf.nn.l2_loss(W1)
			l2_loss += tf.nn.l2_loss(b1)
			l2_loss += tf.nn.l2_loss(W2)
			l2_loss += tf.nn.l2_loss(b2)
			self.f1 = tf.nn.relu(tf.nn.xw_plus_b(self.feat_flat_drop,W1,b1,name='f1'))
			self.f2 = tf.nn.dropout(self.f1,self.dropout_keep_prob)
			# self.y_ = tf.nn.tanh(tf.nn.xw_plus_b(self.f2,W2,b2,name='f2'))
			self.y_ = tf.nn.xw_plus_b(self.f2,W2,b2,name='f2')
		with tf.name_scope('loss'):
			losses = tf.square(self.y_-self.y)
			# self.rmse = tf.sqrt(tf.reduce_mean(losses))
			# self.mae = tf.reduce_mean(tf.abs(self.y_-self.y))
			self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda*l2_loss
		self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		self.sess.run(init)
		writer = tf.summary.FileWriter('./data',self.sess.graph)
		writer.close()
	def partial_fit(self,data):
		feed_dict = {self.x:data['x'],self.y:data['y'],self.dropout_keep_prob:self.keep}
		loss,opt = self.sess.run([self.loss,self.train_step],feed_dict=feed_dict)
		return loss
	def get_random_block_from_data(self,data,batch_size):
		start_idx = np.random.randint(0,len(data['y']))
		x,y = [],[]
		i = start_idx
		while len(x) < batch_size and i < len(data['x']):
			if len(data['x'][i]) == len(data['x'][start_idx]):
				y.append([data['y'][i]])
				x.append(data['x'][i])
				i = i + 1
			else:
				break
		i = start_idx
		while len(x) < batch_size and i >= 0:
			if len(data['x'][i]) == len(data['x'][start_idx]):
				y.append([data['y'][i]])
				x.append(data['x'][i])
				i = i - 1
			else:
				break
		return {'x':x,'y':y}

	def consistent_shuffle(self,a,b):
		rand_state = np.random.get_state()
		np.random.shuffle(a)
		np.random.set_state(rand_state)
		np.random.shuffle(b)
	def train(self,train_data,validation_data,test_data):
		if self.verbose > 0:
			t1 = time()
			init_train = self.partial_evaluate(train_data)
			init_valid = self.partial_evaluate(validation_data)
			print('init: train=%.4f,validation=%.4f [%.1fs]'%(init_train,init_valid,time()-t1))
		for epoch in range(self.epochs):
			t1 = time()
			self.consistent_shuffle(train_data['x'],train_data['y'])
			losses = []
			total_batch = int(len(train_data['y'])/self.batch_size)
			for i in range(total_batch):
				batch_xs = self.get_random_block_from_data(train_data,self.batch_size)
				# print(batch_xs)
				loss = self.partial_fit(batch_xs)
				losses.append(loss)
			
			t2 = time()
			print('epoch:{},loss:{}'.format(epoch+1,np.mean(losses)))
			train_result = self.partial_evaluate(train_data)
			valid_result = self.partial_evaluate(validation_data)
			test_result = self.partial_evaluate(test_data)
			self.train_rmse.append(train_result)
			self.valid_rmse.append(valid_result)
			self.test_rmse.append(test_result)
			if self.verbose > 0 and epoch % self.verbose == 0:
				print("Epoch %d [%.1fs],train=%.4f,validation=%.4f,test=%.4f [%.1fs]"%(epoch+1,t2-t1,train_result,valid_result,test_result,time()-t2))
			if self.eva_termination(self.valid_rmse):
				break
		if self.pretrain_flag < 0:
			print("save model to file as pretrain.")
			self.saver.save(self.sess,self.save_file)
	def eva_termination(self,valid):
		if len(valid) > 5:
			if valid[-1]>=valid[-2] and valid[-2]>=valid[-3] and valid[-3]>=valid[-4] and valid[-4]>=valid[-5]:
				return True
		return False 
	def evaluate(self,data):
		num_samples = len(data['y'])
		feed_dict = {self.x:data['x'],self.y:[ [y] for y in data['y']],self.dropout_keep_prob:1.0}
		y_ = self.sess.run(self.y_,feed_dict=feed_dict)
		y_ = np.reshape(y_,(num_samples,))
		y = np.reshape(data['y'],(num_samples,))
		y_bounded = np.maximum(y_,np.ones(num_samples)*min(y))
		y_bounded = np.minimum(y_,np.ones(num_samples)*max(y))
		# print(y_bounded)
		rmse = np.sqrt(np.mean(np.square(np.subtract(y_bounded,y))))
		return rmse
	def partial_evaluate(self,data):
		num_samples = len(data['y'])
		batch_size = 256
		mse = 0
		for start in range(0,num_samples,batch_size):
			end = start + batch_size
			if start + batch_size > num_samples:
				end = num_samples
			batch_x = data['x'][start:end]
			batch_y = data['y'][start:end]
			feed_dict = {self.x:batch_x,self.y:[ [y] for y in batch_y],self.dropout_keep_prob:1.0}
			y_ = self.sess.run(self.y_,feed_dict=feed_dict)
			y_ = np.reshape(y_,(-1,))
			y = np.reshape(batch_y,(-1,))
			y_bounded = np.maximum(y_,np.ones_like(y)*min(y))
			y_bounded = np.minimum(y_,np.ones_like(y)*max(y))
			mse += np.sum(np.square(np.subtract(y_bounded,y)))
		rmse = np.sqrt(mse/num_samples)
		return rmse
def main():
	args = parse_args()
	dataset = args.dataset
	hfilter_size = eval(args.hfilter_size)
	hfilter_num = eval(args.hfilter_num)
	vfilter_size = eval(args.vfilter_size)
	vfilter_num = eval(args.vfilter_num)
	reg = args.reg
	lr = args.lr
	batch_size = args.batch_size
	epochs = args.epochs
	dropout = args.dropout
	embedding_dim = args.embedding_dim
	ds = DataSet('./data/',dataset)
	valid_dim = 0
	if dataset == 'ml-tag':
		valid_dim = 3
	elif dataset == 'frappe':
		valid_dim = 10
	else:
		valid_dim = 5
	model = CCONN(ds.feat_dim,valid_dim,embedding_dim,hfilter_size,hfilter_num,vfilter_size,vfilter_num,epochs,batch_size,lr,reg,dropout)
	model.train(ds.train_data,ds.validation_data,ds.test_data)

if __name__ == '__main__':
	main()




