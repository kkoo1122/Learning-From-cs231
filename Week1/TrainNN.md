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
  -**Not common to do PCA or whitening**
  
 
  
