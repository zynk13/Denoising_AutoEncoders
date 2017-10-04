import cv2
import tensorflow as tf
from scipy import misc
import matplotlib.image as mpimg
import numpy as np 
import os.path
from StringIO import StringIO
from tensorflow.python.lib.io import file_io

# Basic model parameters as external flags.
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('input_dir', 'input', 'Input Directory.')

# folder = os.path.join(FLAGS.input_dir, 'fenced');

# print folder



# with file_io.read_file_to_string("gs://denoisingbucket/fenced/f_1.png", 'r') as f:
# with file_io.FileIO("gs://denoisingbucket/fenced/f_1.png", 'r') as f:
f = StringIO(file_io.read_file_to_string('gs://denoisingbucket/fenced/dataset_array.npy'))
array=np.load(f)
print array.shape
# f = StringIO(file_io.read_file_to_string('gs://denoisingbucket/fenced/f_1.png'))

# nparr = np.fromstring(f.getvalue(), np.uint8)
# im = cv2.imdecode(nparr,0)
# im=np.array(im, dtype=np.float32)
# im/=255
# print im
# im=cv2.imread(f,0)
# print im
# filename_queue = tf.train.string_input_producer([csv_file])


# im = cv2.imread("/Users/mohitakhakharia/Desktop/MarsWorkSpace/Denoising_AutoEncoders/dataset/gray/fenced/f_1.png",0)
# print type(im)
# im=np.array(im, dtype=np.float32)
# im/=255
# print (im)


# filename_queue = tf.train.string_input_producer(['/Users/mohitakhakharia/Desktop/MarsWorkSpace/Denoising_AutoEncoders/dataset/gray/fenced/f_1.png'])
# reader = tf.WholeFileReader()
# key, value = reader.read(filename_queue)
# my_img = tf.image.decode_png(value)
# print my_img

# cimage=tf.image.convert_image_dtype(my_img, tf.float32)
# print cimage
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#   sess.run(init_op)
#   print cimage.eval()
# image = my_img.eval()
# print(image.shape)

# im22=tf.decode_raw(my_img, tf.float32)
# from google.cloud import storage
# import matplotlib.image as mpimg
# from PIL import Image
# import numpy
# import collections
# from tensorflow.python.framework import dtypes
# import glob
# from pylab import *
# import io


# img = misc.imread('/Users/mohitakhakharia/Desktop/MarsWorkSpace/Denoising_AutoEncoders/dataset/gray/fenced/f_1.png',mode='F')
# im=mpimg.imread('/Users/mohitakhakharia/Desktop/MarsWorkSpace/Denoising_AutoEncoders/dataset/gray/fenced/f_1.png')
# print im


# client = storage.Client()
# bucket = client.get_bucket('denoisingbucket')

# # blob = bucket.get_blob('fenced/f_1.png')
# # print(blob.download_as_string())
# # print(blob)
# # print('Content-type: {}'.format(blob.content_type))

# # im=mpimg.imread(io.BytesIO(blob.download_as_string(),mimetype='image/png'))
# # im=mpimg.imread(io.BytesIO(blob.download_as_string()))
# # print np.array(array(im))
# # print im
# # blobs = bucket.list_blobs()
# # for blob in blobs.:
# #     print(blob.name)

# # blobs = bucket.list_blobs(prefix="fenced/")
# # for prefix in blobs.prefixes:
# #     print(prefix)
# # import logging
# # import os
# # import cloudstorage as gcs
# # import webapp2

# # from google.appengine.api import app_identity


# blobs = bucket.list_blobs(prefix="fenced/", delimiter=None)
# print('Blobs:')
# for blob in blobs:
# 	print(blob.name)