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

