# Tensorflow 2.0
## Code Memo

- Difference between **tf.matmul** and **tf.multiply**
  - tf.matmul: is the real matrix multiplication. Create a matrix
  - tf.multiply: means two matrices multiply each element by the same raw and column. Create a matrix. Two matrices must be **the same size**.
  

## TensorFlow coding example from CS231
### TensoFlow 2.0
```python
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
```
#### Code description
##### Tensorflow Neural Net
- Convert input numpy arrays to TF tensors. Create weights as tf.Variable
```python
x = tf.convert_to_tensor(np.random.randn(N, D), np.float32)
y = tf.convert_to_tensor(np.random.randn(N, D), np.float32)
w1 = tf.Variable(tf.random.uniform((D, H)))
w2 = tf.Variable(tf.random.uniform((D, H)))
```
- Use tf.GradientTape() context to build **dynamic** computation graph
```python
with tf.GradientTape() as tape:
```
- All forward-pass operations in the contexts (including function calls) gets traced for computing gradient later.
<p align="center">
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/TF_NN_EX1.png" alt="drawing" width="200"/>
</p>

```python
 h = tf.maximum(tf.matmul(x, w1), 0)
 y_pred = tf.matmul(h, w2)
 diff = y_pred - y
 loss = tf.reduce_mean(tf.reduce_sum(diff **2, axis=1))
 ```
 - tape.gradient() uses the traced computation graph to compute gradient for the weights
 <p align="center">
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/TF_NN_back.png" alt="drawing" width="200"/>
</p>

 ```python
 gradients = tape.gradient(loss, [w1, w2])
 ```
- **Train the network:** Run the training step over and over, use gradient to update weights
```python
learning_rate = 1e-6
for t in range(50):
    with tf.GradientTape() as tape:
    # Use tf.GradientTape() context to build dynamic computation graph
        h = tf.maximum(tf.matmul(x, w1), 0)
        y_pred = tf.matmul(h, w2)
        diff = y_pred - y
        loss = tf.reduce_mean(tf.reduce_sum(diff **2, axis=1))
    gradients = tape.gradient(loss, [w1, w2])
    w1.assign(w1 - learning_rate * gradients[0])
    w2.assign(w2 - learning_rate * gradients[1])
```
##### Tensorflow Optimizer
- Can use **optimizer** to compute gradients and update weights
```python
optimizer = tf.optimizers.SGD(1e-6)
learning_rate = 1e-6
for t in range(50):
  with ti.GradientTape() as tape:
    ...
    ...
  gradients = tape.gradient(loss, [w1, w2])
  optimizer.apply_gradients(zip(gradients, [w1, w2]))
```
##### Tensorflow Loss
- Use predefined common losses
```python
for t in range(50):
  with ti.GradientTape() as tape:
    ...
    ...
    loss = tf.losses.MeanSquaredError()(y_pred, y)
  gradients = tape.gradient(loss, [w1, w2])
  optimizer.apply_gradients(zip(gradients, [w1, w2]))
```

### Keras: High-Level Wrapper
- Keras is a layer on top of TensorFlow, makes common thins easy to do (Used to be third-party, now merged into TensorFlow)

```python
import numpy as np
import tensorflow as tf

N, D, H = 64, 1000, 100


x = tf.convert_to_tensor(np.random.randn(N, D), np.float32)
y = tf.convert_to_tensor(np.random.randn(N, D), np.float32)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(H, input_shape=(D, ),
                                activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(D))

optimizer =tf.optimizers.SGD(1e-1)

'''
losses = []
for t in range(50):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = tf.losses.MeanSquaredError()(y_pred, y)
    gradients = tape.gradent(
        loss, model.trainable_variables)

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))
'''
#for loop above can  be write as keras code below
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=optimizer)
history = model.fit(x, y, epochs=50, batch_size=N)
```
- Define model as a sequence of layers
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(H, input_shape=(D, ),
                                activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(D))
```

- Get oput by calling the model
```python
y_pred = model(x)
```

- Apply gradient to all trainable variables (weights) in the model
```python
gradients = tape.gradent(
    loss, model.trainable_variables)

optimizer.apply_gradients(
    zip(gradients, model.trainable_variables))
```
