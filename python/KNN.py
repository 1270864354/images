import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_cifar10_batch(cifar10_dataset_folder_path,batch_id):
	with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id),'rb') as f:
		reader = pickle.load(f)
	features = reader['data'].reshape((len(reader['data']),3,32,32)).transpose(0,2,3,1)
	labels = reader['labels']
	return  features,labels

cifar10_path = 'E:/python/image/cifar-10-batches-py'
x_train,y_train = load_cifar10_batch(cifar10_path,1)
for i in range(2,6):
	features,labels = load_cifar10_batch(cifar10_path,i)
	x_train,y_train = np.concatenate([x_train,features]),np.concatenate([y_train,labels])

with open(cifar10_path + '/test_batch','rb') as f:
	reader = pickle.load(f)
	x_test = reader['data'].reshape((len(reader['data']),3,32,32)).transpose(0,2,3,1)
	y_test = reader['labels']

x_train_rows = x_train.reshape(x_train.shape[0],32*32*3)
x_test_rows = x_test.reshape(x_test.shape[0],32*32*3)

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

x_train_rows = minmax.fit_transform(x_train_rows)
x_test_rows = minmax.fit_transform(x_test_rows)

from sklearn.neighbors import KNeighborsClassifier

k = [1,3,5]
for i in k:
	model = KNeighborsClassifier(n_neighbors = i,n_jobs =4)
	model.fit(x_train_rows,y_train)
	preds = model.predict(x_test_rows)
	print 'k = %s, Accuracy = %f' % (i,np.mean(y_test == preds))