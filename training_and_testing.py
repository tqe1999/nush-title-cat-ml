import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import msgpack
from create_features import create_feature_set_and_labels

# Use this if the messagepack has been created: Run create_features.py once to create messagepack
with open('data.msgpack', 'rb') as data_file:
	train_features,train_labels,test_features,test_labels = msgpack.unpack(data_file)
# Use this if straight up runnig training_and_testing.py as it will call the create_feature_set_and_labels in create_features.py
#train_features,train_labels,test_features,test_labels = create_feature_set_and_labels('data_train.csv', 'clarity_train.labels', 'conciseness_train.labels')


# Features are special words
# Labels are [clarity, conciseness]
#Comments
'''
Feed forward
input > weight > hidden layer 1 (activation function) > weights > 
hidden layer 2 (activation function) > weights > output layer

Backprop
compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)

Feed forward + backprop = epoch

'''

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2 # Output layer
batch_size = 100

# height x width
x = tf.placeholder('float', [None, len(train_features[0])]) # data
y = tf.placeholder('float') # label

def neural_network_model(data):

	# (input_data * weights) + biases

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_features[0]), n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases']) # Sum weight
	l1 = tf.nn.relu(l1) # Activation function

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases']) # Sum weight
	l2 = tf.nn.relu(l2) # Activation function

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases']) # Sum weight
	l3 = tf.nn.relu(l3) # Activation function

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases'] # Output layer does not sum

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) # ...Cost?

	# Learning_rate = 0.01

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# Training
		for epoch in range(hm_epochs):
			epoch_loss = 0

			i = 0
			while i < len(train_features):
				start = i
				end = i+batch_size

				batch_x = np.array(train_features[start:end])
				batch_y = np.array(train_labels[start:end])

				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x,y: batch_y})
				epoch_loss += c
				i += batch_size

			print('Epoch', epoch + 1, 'completed out of', hm_epochs,', loss:', epoch_loss)

		# Evaluating correctness
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy:", accuracy.eval({x:test_features, y:test_labels}))


train_neural_network(x)
















