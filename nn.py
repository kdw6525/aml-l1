"""
nn.py

Authors:
Kyle West
Magda Tomazani

Making a NN that takes 8x1 binary input vector. The input will always have exactly one 1 and seven 0s. 
The input maps directly to the class we want the nn to classify as, ex:
input:  < 1, 0, 0, 0, 0, 0, 0, 0 >
output: < 1, 0, 0, 0, 0, 0, 0, 0 >

We want 3 layers:
input layer:  8 nodes, aka 8x1 vector
hidden layer: 3 nodes (but for each class), aka 3x8 + 1 for bias
output layer: 8 nodes, aka 8x1 vector


In this lab you will implement a neural network. The implementation can be made in any language of your choice:
Java, MatLab, Octave, Python, Prolog (for the extra challenge) ... your choice, but you can not (yet) rely on high-level
libraries such as Keras, PyTorch, Theano, Tensorflow, Deeplearning4J, the Matlab Neural Network Toolbox, etc. You
need to implement your own backpropagation for this assignment. You should solve this assignment in pairs. 
(Hint: Things get easier when you vectorise.)

The goal of the implementation is to produce and train a network with 3 layers: input - hidden - output. Both the
input layer and the output layer will have 8 nodes, the hidden layer only 3 nodes (+biases).
The learning examples will each have 7 zeros and 1 one in them (so there will be only 8 different learning examples,
and you will have to repeat them) and the output the network should learn is exactly the same as the input. So when
the input layer is given < 0, 0, 0, 1, 0, 0, 0, 0 > as input, the output to aim for is also < 0, 0, 0, 1, 0, 0, 0, 0 >.
Try to get your network to learn this reproducing function on the 8 different learning examples.
Then study the weights and the activations of the hidden nodes of your network and try to interpret them. Your
deliverables for this week’s assignment are (1) a runnable (compilable and runnable) demo of your network (this
means including any extra files such as learning examples if needed) and (2) a short report where you (i) describe your
software and how to use it (like a README file), (ii) a brief description of the learning performance of your network
(how many examples does it need to converge, how long does that take, does it converge every time or do you have to
be lucky with the parameters1, etc.) and (iii) your interpretation of the learned weights.

"""

import numpy as np

# Literally all versions of input:
# < 1, 0, 0, 0, 0, 0, 0, 0 >
# < 0, 1, 0, 0, 0, 0, 0, 0 >
# < 0, 0, 1, 0, 0, 0, 0, 0 >
# < 0, 0, 0, 1, 0, 0, 0, 0 >
# < 0, 0, 0, 0, 1, 0, 0, 0 >
# < 0, 0, 0, 0, 0, 1, 0, 0 >
# < 0, 0, 0, 0, 0, 0, 1, 0 >
# < 0, 0, 0, 0, 0, 0, 0, 1 >
data = np.array([
    np.array([1, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 1, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 1, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 1, 0, 0]),
    np.array([0, 0, 0, 0, 0, 0, 1, 0]),
    np.array([0, 0, 0, 0, 0, 0, 0, 1]),
])