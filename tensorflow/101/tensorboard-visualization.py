import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(input, n_layer, in_size, out_size, activation_function=None):
    # input: input from last layer
    # in_size: the size for the input information i.e. number of neuron from last layer
    # out_size: hidden neuron for new layer
    # activation_function
    layer_name = "layer_%s" % n_layer
    with tf.name_scope("layers"):
        with tf.name_scope("weights"):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.summary.histogram(layer_name + "/weights", weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
            tf.summary.histogram(layer_name + "/biases", biases)
        with tf.name_scope("inputs"):
            wx_plus_b = tf.matmul(input, weights) + biases
        if activation_function is None:
            output = wx_plus_b
        else:
            output = activation_function(wx_plus_b)
        tf.summary.histogram(layer_name + "/output", output)
        return output


# Make up some real data
x_data = np.linspace(-1, 1, 3000)[:, np.newaxis]  # will give new 300 row with single column in array
noise = np.random.normal(0, 0.05, x_data.shape)  # to make it look like reel data we add noise
y_data = np.square(x_data) - 0.5 + noise

# Visualize the data generated
# plt.scatter(x_data, y_data, s=1, color='b')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Sample Data')
# plt.show()

# define placeholder
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1])  # None is 0 sample and 1 is for number of features as feature is 1
    ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 1, 10, activation_function=tf.nn.relu)

# output layer
prediction = add_layer(l1, 2, 10, 1, activation_function=None)

# error between prediction and real data
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar("loss", loss)
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# important step
init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, i)
