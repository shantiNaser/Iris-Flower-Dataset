
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

import tensorflow as tf
import numpy as np
import time

intial_t = time.time()

def name_encode(name):
	val=[]
	if name == "Iris-setosa":
		val = [1,0,0]
	elif name == "Iris-versicolor":
		val = [0,1,0]
	elif name == "Iris-virginica":
		val = [0,0,1]	
	return val

#this function use for load a data from file in folder 

def data_encode(file):
	X = []
	Y = []
	train_file = open(file, 'r')
	for line in train_file.read().strip().split('\n'):
		line = line.split(',')
		X.append([line[0], line[1], line[2], line[3]])
		Y.append(name_encode(line[4]))
	return X, Y

# here this function use to Defining a Perceptron 
def model(x, weights, bias):
	layer_1 = tf.add(tf.matmul(x, weights["hidden"]), bias["hidden"])
	layer_1 = tf.nn.relu(layer_1)

	output_layer = tf.matmul(layer_1, weights["output"]) + bias["output"]
	return output_layer



train_X , train_Y = data_encode('iris.train')
test_X , test_Y = data_encode('iris.test')



learning_rate = 0.01
trin_t = 2000
display = 200



n_in = 4
n_hidden = 10
n_out = 3


X = tf.placeholder("float", [None, n_in])
Y = tf.placeholder("float", [None, n_out])
		

weights = {
	"hidden" : tf.Variable(tf.random_normal([n_in, n_hidden]), name="weight_hidden"),
	"output" : tf.Variable(tf.random_normal([n_hidden, n_out]), name="weight_output")
}

bias = {
	"hidden" : tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden"),
	"output" : tf.Variable(tf.random_normal([n_out]), name="bias_output")
}	


pred = model(X, weights, bias) 


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, names=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(trin_t):
		_, c = sess.run([optimizer, cost], feed_dict={X: train_X, Y: train_Y})
		if(epoch + 1) % display == 0:
			print "Epoch: ", (epoch+1), "Cost: ", c
	
	print("Finished")
	
	test_result = sess.run(pred, feed_dict={X: test_X})
	correct_pred = tf.equal(tf.argmax(test_result, 1), tf.argmax(train_Y, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
	print "Accuracy:", accuracy.eval({X: test_X, Y: test_Y})


end_time = time.time()

print "Completed in ", end_time - intial_t , " seconds"

	


