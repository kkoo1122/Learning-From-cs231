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
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/Activation_functions.png" alt="drawing" width="200"/>
</p>
- **TLDR:In practice:**
  - Use **ReLU. Be careful with your learning rates
  - Try out **Leaky ReLU / Maxout / ELU**
  - try out **tanh** but don't expect much
  - **Don't use sigmoid**
