import tensorflow as tf
import numpy as np

# ------------Tensorflow structure start---------------
# create data
x_data = np.random.rand(100).astype(np.float32)
# y = mx + c where m = weight and c = bias
y_data = x_data * 0.1 + 0.3

#  create tensorflow structure

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # generate 1 unit between range of -1 and 1
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

# compare the loss between original and calculated value

loss = tf.reduce_mean(tf.square(y - y_data))  # mean of error in 100 samples

# now using optimizer to minimize loss
optimizer = tf.train.GradientDescentOptimizer(0.5)  # learning rate is 0.5
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()  # important
# --------------Tensorflow structure end--------------

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
