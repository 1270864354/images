import pickle
import tensorflow as tf 
import load_test
import numpy as np

with open('/root/work/flower/flower_photos/load_testdata.pkl','rb') as f:
	d = pickle.load(f)
	print 'Load success!'

w = d['w']
h = d['h']
c = d['c']
n_class = d['n_class']
x_train_ = d['x_train_']
y_train_ = d['y_train_']
x_val = d['x_val']
y_val = d['y_val']
x_test = d['x_test']
y_test = d['y_test']
path = d['path']
model_path = d['model_path']



img_shape = x_train_.shape
keep_prob = 0.6
epochs = 20
batch_size = 128

inputs_ = tf.placeholder(tf.float32,[None,w,h,c],name = 'inputs_')
targets_ = tf.placeholder(tf.float32,[None,n_class],name = 'targets_')

conv1 = tf.layers.conv2d(inputs_,64,(2,2),padding='same',activation = tf.nn.relu,
						kernel_initializer = tf.truncated_normal_initializer(mean = 0.0,stddev = 0.1))
conv1 = tf.layers.max_pooling2d(conv1,(2,2),(2,2),padding = 'same')

conv2 = tf.layers.conv2d(conv1, 128, (2,2), padding='same', activation=tf.nn.relu,
                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

conv3 = tf.layers.conv2d(conv2, 256, (2,2), padding='same', activation=tf.nn.relu,
                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')

shape = np.prod(conv3.get_shape().as_list()[1:])
conv3 = tf.reshape(conv3,[-1,shape])

fc1 = tf.contrib.layers.fully_connected(conv3,2048,activation_fn=tf.nn.relu)
fc1 = tf.nn.dropout(fc1,keep_prob)
fc2 = tf.contrib.layers.fully_connected(fc1,1024,activation_fn = tf.nn.relu)

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

