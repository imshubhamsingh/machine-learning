import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(input, in_size, out_size, activation_function=None):
    # input: input from last layer
    # in_size: the size for the input information i.e. number of neuron from last layer
    # out_size: hidden neuron for new layer
    # activation_function
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(input, weights) + biases
    if activation_function is None:
        output = wx_plus_b
    else:
        output = activation_function(wx_plus_b)
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
xs = tf.placeholder(tf.float32, [None, 1])  # None is 0 sample and 1 is for number of features as feature is 1
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# important step
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data, s=1, color='b')
plt.ion()
plt.show()


for i in range(2000):
    # train
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step for improvement
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(ax.lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
        line = ax.plot(x_data, prediction_value, c='r', lw=5)
        plt.pause(0.1)
