# Introduction to Neural Network
Source: http://cs231n.github.io/optimization-2/


- Backpropagation:
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/Screenshot%20from%202019-10-02%2010-35-57.png" alt="drawing" width="500"/>

- Gate pattern of graident flow:
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/gateflow.png" alt="drawing" width="500"/>

  1. **add gate**: always takes the gradient on its output and distributes it equally to all of its inputs, regardless of what their values were during the forward pass. This follows from the fact that the local gradient for the add operation is simply +1.0, so the gradients on all inputs will exactly equal the gradients on the output because it will be multiplied by x1.0 (and remain unchanged). In the example circuit above, note that the + gate routed the gradient of 2.00 to both of its inputs, equally and unchanged.
  
  2. **max gate**: routes the gradient. Unlike the add gate which distributed the gradient unchanged to all its inputs, the max gate distributes the gradient (unchanged) to exactly one of its inputs (the input that had the highest value during the forward pass). This is because the local gradient for a max gate is 1.0 for the highest value, and 0.0 for all other values. In the example circuit above, the max operation routed the gradient of 2.00 to the **z** variable, which had a higher value than **w**, and the gradient on **w** remains zero.
  
  3. **multiply gate**: is a little less easy to interpret. Its local gradients are the input values (except switched), and this is multiplied by the gradient on its output during the chain rule. In the example above, the gradient on x is -8.00, which is -4.00 x 2.00.


- Recap of derivative:
<img src="https://github.com/kkoo1122/Learning-From-cs231/blob/master/image/Screenshot%20from%202019-10-02%2010-23-46.png" alt="drawing" width="500"/>


## Summary
- (Fully-connected) Neural Networks are stacks of linear functions and
nonlinear activation functions; they have much more representational
power than linear classifiers
- backpropagation = recursive application of the chain rule along a
computational graph to compute the gradients of all
inputs/parameters/intermediates
- implementations maintain a graph structure, where the nodes implement
the forward() / backward() API
- forward: compute result of an operation and save any intermediates
needed for gradient computation in memory
- backward: apply the chain rule to compute the gradient of the loss
function with respect to the inputs
