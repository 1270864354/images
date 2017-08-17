import glob
import os
import tensorflow as tf 
import numpy as np 
from PIL import Image
#数据集地址
path = '/root/work/flower/flower_photos'
#模型保存地址
model_path='/root/work/flower/flower_photos'

#64*64*3
w = 64
h =64
c = 3
#读取图片
def  read_img(path):
	cate = [path + "/" + x for x in os.listdir(path) if os.path.isdir(path+"/" + x)]
	imgs = []
	labels =[]
	for idx,folder in enumerate(cate):
		for im in glob.glob(folder +'/'+'*.jpg'):
			img = Image.open(im)
			img = img.resize((w,h))
			img_px = np.asarray(img,dtype = 'float32')
			imgs.append(img_px)
			labels.append(idx)
	print "Load finish!"
	return np.asarray(imgs,np.float32),np.asarray(labels,np.float32)
data,label =read_img(path)

def read_one_img(path):
	imgs = []
	labels =[]
	img = Image.open(path)
	img = img.resize((w,h))
	img_px = np.asarray(img,dtype = 'float32')
	imgs.append(img_px)
	labels.append(idx)
	return np.asarray(imgs,np.float32),np.asarray(labels,np.float32)



num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label =label[arr]

#将data进行最大最小归一化
from sklearn.preprocessing  import MinMaxScaler
minmax = MinMaxScaler()

data_rows = data.reshape(data.shape[0],w*h*c)
data = minmax.fit_transform(data_rows)
data =data.reshape(data.shape[0],w,h,c)

from sklearn.preprocessing import LabelBinarizer
n_class = 5
lb = LabelBinarizer().fit(np.array(range(n_class)))
label = lb.transform(label)


#将所有数据分为训练集、验证集和测试集
validation_ratio = 0.6
test_ratio=0.8
s1=np.int(num_example*validation_ratio)
s2 = np.int(num_example*test_ratio)
x_train_=data[:s1]
y_train_=label[:s1]
x_val = data[s1:s2]
y_val = label[s1:s2]
x_test=data[s2:]
y_test=label[s2:]


img_shape = x_train_.shape
keep_prob = 0.6
epochs = 20
batch_size = 64

inputs_ = tf.placeholder(tf.float32,[None,w,h,c],name = 'inputs_')
targets_ = tf.placeholder(tf.float32,[None,n_class],name = 'targets_')

conv1 = tf.layers.conv2d(inputs_,64,(2,2),padding='same',activation = tf.nn.relu,
						kernel_initializer = tf.truncated_normal_initializer(mean = 0.0,stddev = 0.1))
conv1 = tf.layers.max_pooling2d(conv1,(2,2),(2,2),padding = 'same')

conv2 = tf.layers.conv2d(conv1, 128, (4,4), padding='same', activation=tf.nn.relu,
                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

shape = np.prod(conv2.get_shape().as_list()[1:])
conv2 = tf.reshape(conv2,[-1,shape])

fc1 = tf.contrib.layers.fully_connected(conv2,1024,activation_fn=tf.nn.relu)
fc1 = tf.nn.dropout(fc1,keep_prob)
fc2 = tf.contrib.layers.fully_connected(fc1,512,activation_fn = tf.nn.relu)

logits_ = tf.contrib.layers.fully_connected(fc2,n_class,activation_fn = None)
logits_ = tf.identity(logits_,name='logits_')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits_,labels =targets_))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits_,1),tf.argmax(targets_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name = 'accuracy')

save_model_path = model_path + '/train'
count = 0
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(epochs):
		for batch_i in range(img_shape[0]//batch_size -1):
			feature_batch = x_train_[batch_i * batch_size:(batch_i +1)*batch_size]
			labels_batch = y_train_[batch_i * batch_size: (batch_i+1)*batch_size]
			train_loss, _ = sess.run([cost, optimizer],
                                    feed_dict={inputs_: feature_batch,
                                               targets_: labels_batch})
			val_acc = sess.run(accuracy,
                            feed_dict={inputs_: x_val,
                                        targets_: y_val})
			if(count%10==0):
				print 'Epoch {:>2}, Train Loss {:.4f}, Validation Accuracy {:4f} '.format(epoch + 1, train_loss, val_acc)

			count += 1
	saver = tf.train.Saver()
	save_path = saver.save(sess,save_model_path)

import random

loaded_graph = tf.Graph()
test_batch_size= 10
with tf.Session(graph=loaded_graph) as sess:
	loader = tf.train.import_meta_graph(save_model_path + '.meta')
	loader.restore(sess, save_model_path)
	loaded_x = loaded_graph.get_tensor_by_name('inputs_:0')
	loaded_y = loaded_graph.get_tensor_by_name('targets_:0')
	loaded_logits = loaded_graph.get_tensor_by_name('logits_:0')
	loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

	test_batch_acc_total = 0
	test_batch_count = 0

	print "Begin test..."
	for batch_i in range(x_test.shape[0]//test_batch_size-1):
		test_feature_batch = x_test[batch_i * test_batch_size: (batch_i+1)*test_batch_size]
		test_label_batch = y_test[batch_i * test_batch_size: (batch_i+1)*test_batch_size]
		test_batch_acc_total += sess.run(
			loaded_acc,
			feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch})
		test_batch_count += 1

	print 'Test Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count)