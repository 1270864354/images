import numpy as np
import tensorflow as tf
import pickle
import warnings
import os
os.chdir('/root/images/cifar-10-batches-py')
warnings.filterwarnings('ignore')


def load_cifar10_batch(cifar10_dataset_folder_path,batch_id):
	with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id),'rb') as f:
		reader = pickle.load(f)
	features = reader['data'].reshape((len(reader['data']),3,32,32)).transpose(0,2,3,1)
	labels = reader['labels']
	return  features,labels

cifar10_path = '/root/images/cifar-10-batches-py'


x_train,y_train = load_cifar10_batch(cifar10_path,1)
for i in range(2,6):
	features,labels = load_cifar10_batch(cifar10_path,i)
	x_train,y_train = np.concatenate([x_train,features]),np.concatenate([y_train,labels])

with open(cifar10_path + '/test_batch','rb') as f:
	reader = pickle.load(f)
	x_test = reader['data'].reshape((len(reader['data']),3,32,32)).transpose(0,2,3,1)
	y_test = reader['labels']

x_train = x_train[0:5000]
y_train = y_train[0:5000]
x_test = x_test[0:1000]
y_test = y_test[0:1000]



from sklearn.preprocessing  import MinMaxScaler
minmax = MinMaxScaler()

x_train_rows = x_train.reshape(x_train.shape[0],32*32*3)
x_test_rows = x_test.reshape(x_test.shape[0],32*32*3)

x_train = minmax.fit_transform(x_train_rows)
x_test = minmax.fit_transform(x_test_rows)

x_train = x_train.reshape(x_train.shape[0],32,32,3)
x_test = x_test.reshape(x_test.shape[0],32,32,3)


from sklearn.preprocessing import LabelBinarizer
n_class = 10
lb = LabelBinarizer().fit(np.array(range(n_class)))
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

from sklearn.model_selection import train_test_split
train_ratio = 0.8
x_train_,x_val,y_train_,y_val = train_test_split(x_train,y_train,train_size = train_ratio,random_state = 123)


img_shape = x_train_.shape
keep_prob = 0.6
epochs = 20
batch_size = 128

inputs_ = tf.placeholder(tf.float32,[None,32,32,3],name = 'inputs_')
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

logits_ = tf.contrib.layers.fully_connected(fc2,10,activation_fn = None)
logits_ = tf.identity(logits_,name='logits_')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits_,labels =targets_))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits_,1),tf.argmax(targets_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name = 'accuracy')

save_model_path = './test_cifar'
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



