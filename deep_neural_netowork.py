import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
start_time = time.time()

import numpy as np
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data")   
LOG_DIR = '/tmp/log'



image_dim = 28    
input_layer =   pow(28,2)       
no_hidden_layer1 = 500
no_hidden_layer2 = 120
no_output_layer  = 10
x_train = tf.placeholder(tf.float32,(None,input_layer),name="x_train")
y_train = tf.placeholder(tf.int32,(None),name="y_train")
learning_rate = 0.02
epochs = 250
batch_size = 60
n_batches = int (mnist.train.num_examples / batch_size)

train_images=mnist.train.images
train_labels=mnist.train.labels
test_images= mnist.test.images
test_labels= mnist.test.labels
def dnn(no_neurons,x_train,activation=True):
    with tf.name_scope("neural_network"):
        input_count = x_train.shape[1].value      # no. of nodes from previous layer 
        std_dev = 2/pow(input_count,0.5)
        initial_wt = tf.truncated_normal([input_count,no_neurons],stddev=std_dev)
        weights = tf.Variable(initial_wt)
        biases = tf.Variable(tf.zeros([no_neurons]))
        result_matrix = tf.matmul(x_train,weights) + biases
        if(activation):
            return tf.nn.relu(result_matrix)
        else:
            return result_matrix

with tf.name_scope("deep_neural_network"):
    hidden_1 = dnn(no_hidden_layer1,x_train,True)
    hidden_2 = dnn(no_hidden_layer2,hidden_1,True)
    logits   = dnn(no_output_layer,hidden_2, False)

with tf.name_scope("loss"):
    loss =tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_train, logits=logits))

with tf.name_scope("evaluation"):
    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits,y_train,1),tf.float32))

with tf.name_scope("training"):
    training = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

X_batch, Y_batch = mnist.train.next_batch(batch_size)

#x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
accuracy_train_=[]
accuracy_test_=[]
init=tf.global_variables_initializer()
writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        if epoch%50==0:
          print(accuracy_train_,"\n", accuracy_test_)
          accuracy_train_,  accuracy_test_=[],[]
          
        for batch_index in range(n_batches):
            X_batch,y_batch = mnist.train.next_batch(batch_size)
            sess.run(training,feed_dict={x_train: X_batch,y_train:y_batch})
       
        accuracy_train = accuracy.eval(feed_dict={x_train: train_images,y_train: train_labels})
        accuracy_train_.append(accuracy_train)
        accuracy_test = accuracy.eval(feed_dict={x_train: test_images,y_train:test_labels})
        accuracy_test_.append(accuracy_test)
      

        print("Epoch {}: Training Accuracy: {},  Testing accuracy: {}".format(epoch,accuracy_train,accuracy_test))
end_time=time.time()
print("--- Total time ---", end_time-start_time)
writer.close()

print(accuracy_train_,"\n", accuracy_test_)
