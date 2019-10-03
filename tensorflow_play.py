import numpy as np
import tensorflow as tf
N, D, H = 64, 1000, 100

# Convert input numpy arrays to TF tensors. Create weights as tf.Variable
x = tf.convert_to_tensor(np.random.randn(N, D), np.float32)
y = tf.convert_to_tensor(np.random.randn(N, D), np.float32)
w1 = tf.Variable(tf.random.uniform((D, H)))
w2 = tf.Variable(tf.random.uniform((D, H)))

optimizer = tf.optimizers.SGD(1e-6)
learning_rate = 1e-6
for t in range(50):
    with tf.GradientTape() as tape:
    # Use tf.GradientTape() context to build dynamic computation graph
        h = tf.maximum(tf.matmul(x, w1), 0)
        y_pred = tf.matmul(h, w2)
        diff = y_pred - y
        #loss = tf.reduce_mean(tf.reduce_sum(diff **2, axis=1))
        loss = tf.losses.MeanSquaredError()(y_pred, y)
    gradients = tape.gradient(loss, [w1, w2])
    optimizer.apply_gradients(zip(gradients, [w1, w2]))
    #w1.assign(w1 - learning_rate * gradients[0])
    #w2.assign(w2 - learning_rate * gradients[1])