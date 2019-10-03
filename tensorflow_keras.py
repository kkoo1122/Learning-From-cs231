import numpy as np
import tensorflow as tf
import timeit

N, D, H = 64, 1000, 100


x = tf.convert_to_tensor(np.random.randn(N, D), np.float32)
y = tf.convert_to_tensor(np.random.randn(N, D), np.float32)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(H, input_shape=(D, ),
                                activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(D))

optimizer =tf.optimizers.SGD(1e-1)

losses = []

'''
for t in range(50):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = tf.losses.MeanSquaredError()(y_pred, y)
    gradients = tape.gradent(
        loss, model.trainable_variables)

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))
'''
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=optimizer)
history = model.fit(x, y, epochs=50, batch_size=N)


'''
@tf.function
def model_static(x, y):
    y_pred = model(x)
    loss = tf.losses.MeanSquaredError()(y_pred, y)
    return y_pred, loss

def model_dynamic(x, y):
    y_pred = model(x)
    loss = tf.losses.MeanSquaredError()(y_pred, y)
    return y_pred, loss


print("static graph:",
    timeit.timeit(lambda: model_static(x,y), number=10))

print("dynamic graph:",
    timeit.timeit(lambda: model_dynamic(x,y), number=10))
'''