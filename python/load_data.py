import glob
import os
import numpy as np 
from PIL import Image
from sklearn.preprocessing  import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer




def read_img(path):
	dict_label = {}
	cate = [path + "/" + x for x in os.listdir(path) if os.path.isdir(path+"/" + x)]
	imgs = []
	labels =[]
	count = 0 
	for idx,folder in enumerate(cate):
		count += 1
		for im in glob.glob(folder +'/'+'*.jpg'):
			class_ = folder.split('/')[-1]
			if class_ not in dict_label.values():
				dict_label[idx] = class_
			img = Image.open(im)
			img = img.resize((w,h))
			img_px = np.asarray(img,dtype = 'float32')
			imgs.append(img_px)
			labels.append(idx)
	return np.asarray(imgs,np.float32),np.asarray(labels,np.float32),dict_label,count
	
def read_one_img(path,w,h):
	imgs = []
	img = Image.open(path)
	img = img.resize((w,h))
	img_px = np.asarray(img,dtype = 'float32')
	imgs.append(img_px)
	return np.asarray(imgs,np.float32)

def disrupt(data,label):
	num_example = data.shape[0]
	arr = np.arange(num_example)
	np.random.shuffle(arr)
	data = data[arr]
	label =label[arr]
	return data,label



def min_max_scaler_fit(data,w,h,c):
	minmax = MinMaxScaler()
	data_rows = data.reshape(data.shape[0],w*h*c)
	data = minmax.fit_transform(data_rows)
	data =data.reshape(data.shape[0],w,h,c)
	return data,minmax


def min_max_scaler_transform(data,minmax,w,h,c):
	data_rows = data.reshape(data.shape[0],w*h*c)
	data = minmax.transform(data_rows)
	data = data.reshape(data.shape[0],w,h,c)
	return data

def one_hot(label,n_class):
	lb = LabelBinarizer().fit(np.array(range(n_class)))
	label = lb.transform(label)
	return label,lb



def split_train_val_test(data,label,validation_ratio,test_ratio):
	s1=np.int(data.shape[0]*validation_ratio)
	s2 = np.int(data.shape[0]*test_ratio)
	x_train_=data[:s1]
	y_train_=label[:s1]
	x_val = data[s1:s2]
	y_val = label[s1:s2]
	x_test=data[s2:]
	y_test=label[s2:]
	return x_train_,y_train_,x_val,y_val,x_test,y_test
	


if __name__ == '__main__':
	import pickle
	#data_path
	path = '/python/images/images'  	#图片地址
	#model_path
	model_path='/python/images/python'   #模型地址
	#64*64*3  w：宽 h:高  c:1 黑白   3 rgb
	w = 64
	h =64
	c = 3
	data,label,dict_label,n_class = read_img(path)
	data,label = disrupt(data,label)
	data,minmax = min_max_scaler_fit(data,w,h,c)
	label,lb = one_hot(label,n_class)
	#validation_ratio and test_ratio
	validation_ratio = 0.6
	test_ratio=0.8
	x_train_,y_train_,x_val,y_val,x_test,y_test = split_train_val_test(data,label,validation_ratio,test_ratio)
	d_data = {}
	d_model = {}
	d_data['x_train_'] = x_train_
	d_data['y_train_'] = y_train_
	d_data['x_val'] = x_val
	d_data['y_val'] = y_val
	d_data['x_test'] = x_test
	d_data['y_test'] = y_test
	d_data['path'] = path
	d_data['model_path'] = model_path
	d_data['w'] = w
	d_data['h'] = h
	d_data['c'] = c
	d_data['n_class'] = n_class
	d_data['dict_label'] = dict_label
	d_model['path'] = path
	d_model['model_path'] = model_path
	d_model['minmax'] = minmax
	d_model['lb'] = lb
	d_model['w'] = w
	d_model['h'] = h
	d_model['c'] = c
	d_model['n_class'] = n_class
	d_model['dict_label'] = dict_label

	
	with open(model_path +'/load_data.pkl', 'wb') as f:
		pickle.dump(d_data,f)
	with open(model_path +'/load_model.pkl', 'wb') as f:
		pickle.dump(d_model,f)
	print 'Load data finished!'