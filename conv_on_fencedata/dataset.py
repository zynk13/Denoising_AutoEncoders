"""A generic module to read data."""
import numpy
import collections
from tensorflow.python.framework import dtypes
# import glob
import io
# from pylab import *
import numpy as np 
# from PIL import Image
# import cv2
# from array import array
# import matplotlib.image as mpimg
# from google.cloud import storage
from StringIO import StringIO
from tensorflow.python.lib.io import file_io


# client = storage.Client()
# bucket = client.get_bucket('denoisingbucket')


class DataSet(object):
    """Dataset class object."""

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float64,
                 reshape=True):
        """Initialize the class."""
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                images.shape[1] * images.shape[2])

        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


def read_data_sets(fake_data=False, one_hot=False,
                        dtype=dtypes.float64, reshape=True,
                        validation_size=5000):
    """Set the images and labels."""
    print "Entry - read_data_sets"
    #Mnist
    # num_training = 12000
    # num_validation = 1000
    # num_test = 3384

    num_training = 1000
    num_validation = 400
    num_test = 171

    # all_images = []
    # cloud_array_file = StringIO(file_io.read_file_to_string('gs://denoisingbucket/fenced/dataset_array.npy'))
    all_images=np.load("/Users/mohitakhakharia/Desktop/MarsWorkSpace/Denoising_AutoEncoders/conv_on_fencedata/dataset_array.npy")

    # all_images=np.load(cloud_array_file)
    print all_images.shape
    print "......."
    train_labels_original = np.zeros(1571)
    all_labels = numpy.asarray(range(0, len(train_labels_original)))
    all_labels = dense_to_one_hot(all_labels, len(all_labels))

    mask = range(num_training)
    train_images = all_images[mask]
    train_labels = all_labels[mask]

    mask = range(num_training, num_training + num_validation)
    validation_images = all_images[mask]
    validation_labels = all_labels[mask]

    mask = range(num_training + num_validation, num_training + num_validation + num_test)
    test_images = all_images[mask]
    test_labels = all_labels[mask]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images, validation_labels, dtype=dtype,
        reshape=reshape)

    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
    ds = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

    return ds(train=train, validation=validation, test=test)



def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot