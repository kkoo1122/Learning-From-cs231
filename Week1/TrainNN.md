# Training Neural Networks

## Overview
1. **One time setup**: 
  - activation functions, preprocessing, weight initialization, regularization, gradient checking
2. **Training dynamics**:
  - babysitting the learning process, paameter updates, hyperparameter optimization
3. **Evaluation**:
  - model ensembles, test-time augmentation

## Activation Functions
<p align="center">
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/Activation_functions.png" alt="drawing" width="600"/>
</p>

- **TLDR:In practice:**
  - Use **ReLU. Be careful with your learning rates
  - Try out **Leaky ReLU / Maxout / ELU**
  - try out **tanh** but don't expect much
  - **Don't use sigmoid**

## Data Preprocessing
- **TLDR: In practice for Images:** center only
e.g. consider CIFAR-10 example with [32,32,3] images
  - Subtract the mean image (e.g. AlexNet) (mean image = [32,32,3] array)
  - Subtract per-channel mean (e.g. VGGNet) (mean along each channel = 3 numbers)
  - Subtract per-channel mean and Divide by per-channel std (e.g ResNet) (mean along each channel = 3 numbers)
  - **Not common to do PCA or whitening**
  
## Weight Initialization
- "Xavier" Initialization: ok in tanh but not ReLU.
```python
dims = [4096] * 7
hs = []
x = np.random.randn(16, dims[0])
for Din, Dout in zip(dims[:-1], dims[1:]):
  W = np.random.randn(Din, Dout) / np.sqrt(Din) #"Xavier initialization: std = 1/sqrt(Din)
  x = np.tanh(x.dot(W))
  hs.append(x)
```

-  In ReLU:
```python
W = np.random.randn(Din, Dout) * np.sqrt(2/Din)
```

## Batch Normalization
- "you want zero0mean unit-variance activation? just make them so."
- consider a batch of activations at some layers. To make each dimension zero-mean unit-variance, apply: 

<p align="center">
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/batch_normal_concept.png" alt="drawing" width="600"/>
</p>

- Makes deep networks **much** easier to train!
- Improves gradient flow
- Allows higher learning rates, faster convergence
- Networks become more robust to initialization
- Acts as regularization during training
- Zero overhead at test-time: can be fused with conv!
- **Behaves differently during training and testing: this is a very common source of bugs**

- Other Normalizations
<p align="center">
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/concept_of_normalizations.png" alt="drawing" width="600"/>
</p>

