### Denoising Autoencoder - Simple Neural Network
A simple neural network that takes in 28x28 noisy mnist digits as inputs, encodes them onto two hidden layers(256 nodes) and recreates the clear output.

## Execution
```
# Command to run
 python autoencoder_simple_on_mnist.py 

# COST - Mean Square
 cost = tf.reduce_mean(tf.pow(y_pred - y, 2)) 
     
# OPTIMIZER 
 optmizer = tf.train.RMSPropOptimizer(0.01).minimize(cost) 
```
##
Images after training the network for 100 epochs

![Denoising AutoEncoder - Simple](/type1_simple_on_mnist/output/100_original_input.png?raw=true "Denoising AutoEncoder - Simple")
![Denoising AutoEncoder - Simple](/type1_simple_on_mnist/output/100_noisy_input.png?raw=true "Denoising AutoEncoder - Simple")
![Denoising AutoEncoder - Simple](/type1_simple_on_mnist/output/100_reconstructed.png?raw=true "Denoising AutoEncoder - Simple")
