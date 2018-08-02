import tensorflow as tf 
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("training_data.pkl", "rb") as f:
	data = pickle.load(f)


x_train = data["x_train"]
y_train = data["y_train"]


# Training param
lr = 0.001
training_step = 10000
batch_size = 50 
display_step = 200

# Network param
num_input = len(x_train[0])
timesteps = len(x_train)
num_hidden = 128
num_classes = 2**2 # max number of bits

# tf graph input
X = tf.placeholder("float", [None, timesteps, num_input])

batchX_placeholder = tf.placeholder(tf.int32, [batch_size, lenx])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, leny])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

input_series = tf.unpack(batchX_placeholder, axis=1)
labels_series = tf.unpack(batchY_placeholder, axis=1)


# Forward pass
current_state = init_state
states_series = []

for current_input in input_series:
	current_input = tf.reshape(current_input, [batch_size, 1])
	input_state_concat = tf.concat(1, [current_input, current_state])

	next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)
	states_series.append(next_state)
	current_state = next_state
