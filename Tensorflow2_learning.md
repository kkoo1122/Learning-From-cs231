# Tensorflow 2.0
## Code Memo

- Difference between **tf.matmul** and **tf.multiply**
  - tf.matmul: is the real matrix multiplication. Create a matrix
  - tf.multiply: means two matrices multiply each element by the same raw and column. Create a matrix. Two matrices must be **the same size**.
  

## TensorFlow coding example froma CS231

```python
import numpy as np
import tensorflow as tf

N, D, H = 64, 1000, 100
```

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
 y_pred = tf.matmul(hm w2)
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

