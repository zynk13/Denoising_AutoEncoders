### Denoising Autoencoder - Simple Neural Network
A simple convolutional neural network that takes in 28x28 noisy mnist digits as inputs, encodes them onto two hidden layers(256 nodes) and recreates the clear output.

## Hyper Parameters
- Kernel Size = 5
- Batch Size = 128
- Number of epochs = 5
- n1 = 16 
- n2 = 32 
- n3 = 64 
- learning_rate = 0.001 

## Execution
```
# Command to run
 python autoencoder_convolutional_on_mnist.py

# COST - Mean Square
 cost = tf.reduce_sum(tf.square(cae(x, weights, biases, keepprob)- tf.reshape(y, shape=[-1, 28, 28, 1])))     
# OPTIMIZER 
 optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)
```
## Images after training the network for 05 epochs

![Denoising AutoEncoder - Convolutional](/type2_convolutional_on_mnist/output/type2_convolutional_on_mnist_output.png?raw=true "Denoising AutoEncoder - Convolutional")