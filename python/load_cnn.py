import glob
import os
import tensorflow as tf 
import numpy as np 
from PIL import Image
from sklearn.preprocessing  import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
import pickle
import load_data

def image_one_label(path):
	with open('/python/images/python/load_model.pkl','rb') as f:
	    dict_model = pickle.load(f)
	model_path =dict_model['model_path']
	minmax =dict_model['minmax']
	lb =dict_model['lb']
	w =dict_model['w']
	h =dict_model['h']
	c =dict_model['c']
	n_class =dict_model['n_class']
	dict_label =dict_model['dict_label']
	x_test = load_data.read_one_img(path,w,h)
	x_test = load_data.min_max_scaler_transform(x_test,minmax,w,h,c)

	loaded_graph = tf.Graph()
	test_batch_size= 1
	with tf.Session(graph=loaded_graph) as sess:
		loader = tf.train.import_meta_graph(model_path + '/train.meta')
		loader.restore(sess, model_path + '/train')
		loaded_x = loaded_graph.get_tensor_by_name('inputs_:0')
		loaded_y = loaded_graph.get_tensor_by_name('targets_:0')
		loaded_logits = loaded_graph.get_tensor_by_name('logits_:0')
		loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')


		print "Begin test..."
		test_feature_batch = x_test[0:1]
		l = sess.run(
			loaded_logits,
			feed_dict={loaded_x: test_feature_batch})
		label = dict_label[l.argmax()]
		return label
