import numpy
import collections
from tensorflow.python.framework import dtypes
import glob
from pylab import *
import matplotlib.image as mpimg
from tempfile import TemporaryFile

def read_data_sets(train_dir, fake_data=False, one_hot=False,
                        dtype=dtypes.float64, reshape=True,
                        validation_size=5000):
    """Set the images and labels."""
    num_training = 12000
    num_validation = 1000
    num_test = 3384

    all_images = []
    for filename in glob.glob('/Users/mohitakhakharia/Desktop/MarsWorkSpace/Denoising_AutoEncoders/dataset/gray/fenced/*.png'):
    
        im=mpimg.imread(filename)
        all_images.append(np.array(array(im)))
        all_images=np.array(all_images)
        all_images = all_images.reshape(all_images.shape[0],all_images.shape[1], all_images.shape[2], 1)
        print all_images.shape
        print "......."    
        np.save('dataset_array.npy', all_images)

read_data_sets("",one_hot=True)