# Convolution Neural Network

## Fully Connected Layer
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/fully_connected_layer.png" alt="drawing" width="500"/>

---
## Convolution Layer
- The filter size should be:
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/cnn_window_size.png" alt="drawing" width="500"/>

Otherwise, it won't fit the image.

### Exercise Time:

- **Recap**:
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/cnn_pic.png" alt="drawing" width="500"/>

- **Ex1**:
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/cnn_ex1.png" alt="drawing" width="500"/>

- **Ex2**:
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/cnn_ex2.png" alt="drawing" width="500"/>


### Summary:
- Accepts a colume of size **W_1 x H_1 x D_1**
- Requires four hyperparameters:
  - Number of filters **K**,
  - their spatial extent **F**,
  - the stride **S**,
  - the amount of zero padding **P**.
- Produces a volume of size **W_2 x H_2 x D_2** where:
  - W_2 = (W_1 - F + 2P)/S + 1
  - H_2 = (H_1 - F + 2P)/S + 1  (i.e. width and height are computed wqally by symmetry)
  - D_2 = K (number of fiters)
- With parameter sharing, it introduces **FxFxD_1** weights per filter, for a total of (**FxFxD_1) x K** weights and **K** biases.
- In the ouput volume, the **d**-th depth slice (of size **W_2 x H_2**) id the result of performing a valid convolution of the **d**-th filter over the input volume with a stride of **S**, and then offset by **d**-th bias.
---
## Pooling Layer
- makes the representations smaller and more manageable.
- operates over each activation map independently.
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/pooling.png" alt="drawing" width="300"/>
