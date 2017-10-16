import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 
import matplotlib.pyplot as plt 
import sys

# NETWORK ARCHITECTURE PARAMETERS AND HYPERPARAMETERS
n_input    = 784  
n_hidden_1 = 256  
n_hidden_2 = 256  
n_output   = 784 
 
epochs     = 110 
batch_size = 100 
disp_step  = 10 
 
#Plot function - The function that shows the comparison between original and noisy image.
#Uncomment the fig to save the images in your disk
def plotresult(org_vec,noisy_vec,out_vec,epoch): 
    
    origimg   = np.reshape(org_vec, (28, 28))
    plt.matshow(origimg, cmap=plt.get_cmap('gray')) 
    plt.title("Original Image") 
    plt.colorbar() 
    # fig1 = plt.gcf()
    # fig1.savefig(str(epoch)+'_original.png', dpi=100)

    noisyimg   = np.reshape(noisy_vec, (28, 28))
    plt.matshow(noisyimg, cmap=plt.get_cmap('gray')) 
    plt.title("Input Image") 
    plt.colorbar() 
    # fig2 = plt.gcf()
    # fig2.savefig(str(epoch)+'_noisy_input.png', dpi=100)
    
    outimg   = np.reshape(out_vec, (28, 28))
    plt.matshow(outimg, cmap=plt.get_cmap('gray')) 
    plt.title("Reconstructed Image") 
    plt.colorbar() 
    # fig3 = plt.gcf()
    # fig3.savefig(str(epoch)+'_reconstructed.png', dpi=100)
    plt.show()
    plt.draw()
    # sys.exit()
 

def main(): 
    mnist = input_data.read_data_sets('data/', one_hot=True) 
    trainimg   = mnist.train.images 
    trainlabel = mnist.train.labels 
    testimg    = mnist.test.images 
    testlabel  = mnist.test.labels 
    print ("MNIST LOADED") 
     
    # PLACEHOLDERS 
    x = tf.placeholder("float", [None, n_input]) 
    y = tf.placeholder("float", [None, n_output]) 
    dropout_keep_prob = tf.placeholder("float") 
     
    # WEIGHTS 
    weights = { 
        'hidden1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), 
        'hidden2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), 
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_output])) 
    } 
    biases = { 
        'bias1': tf.Variable(tf.random_normal([n_hidden_1])), 
        'bias2': tf.Variable(tf.random_normal([n_hidden_2])), 
        'out': tf.Variable(tf.random_normal([n_output])) 
    } 
     
    encode_in = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['hidden1']),biases['bias1']))  
    encode_out = tf.nn.dropout(encode_in, dropout_keep_prob)  
     
    decode_in = tf.nn.sigmoid(tf.add(tf.matmul(encode_out, weights['hidden2']),biases['bias2']))  
    decode_out = tf.nn.dropout(decode_in,dropout_keep_prob)  
     
     
    y_pred = tf.nn.sigmoid(tf.matmul(decode_out,weights['out']) +biases['out']) 

    # COST - Mean Square
    cost = tf.reduce_mean(tf.pow(y_pred - y, 2)) 
     
    # OPTIMIZER 
    optmizer = tf.train.RMSPropOptimizer(0.01).minimize(cost) 
     
    # INITIALIZER 
    init = tf.global_variables_initializer()  
     
    # Launch the graph 
    with tf.Session() as sess: 
        sess.run(init) 
        print ("Start Training") 
        for epoch in range(epochs): 
            num_batch  = int(mnist.train.num_examples/batch_size) 
            total_cost = 0. 
            for i in range(num_batch): 
                batch_xs, batch_ys = mnist.train.next_batch(batch_size) 
                batch_xs_noisy = batch_xs+ 0.3*np.random.randn(batch_size, 784) 
                feeds = {x: batch_xs_noisy, y: batch_xs, dropout_keep_prob: 0.8} 
                sess.run(optmizer, feed_dict=feeds) 
                total_cost += sess.run(cost, feed_dict=feeds) 
            # DISPLAY 
            if epoch % disp_step == 0: 
                print ("Epoch %02d/%02d average cost: %.6f"  % (epoch, epochs, total_cost/num_batch)) 
                print ("Start Test") 
                randidx   = np.random.randint(testimg.shape[0], size=1) 
                orgvec    = testimg[randidx, :] 
                testvec   = testimg[randidx, :] 
                label     = np.argmax(testlabel[randidx, :], 1) 
     
                print ("Test label is %d" % (label))  
                noisyvec = testvec + 0.3*np.random.randn(1, 784) 
                outvec   = sess.run(y_pred,feed_dict={x: noisyvec,dropout_keep_prob: 1}) 
     
                plotresult(orgvec,noisyvec,outvec,epoch) 
                print ("restart Training") 

if __name__ == '__main__':
  main()
