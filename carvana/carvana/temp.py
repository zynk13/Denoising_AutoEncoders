import numpy
import collections
# from tensorflow.python.framework import dtypes
# import glob
import matplotlib.pyplot as plt
import io
# from pylab import *
import numpy as np 
from PIL import Image
# import cv2
import os
# from array import array
import scipy
from scipy.misc import toimage
import matplotlib
import matplotlib.image as mpimg
from skimage import io, exposure, img_as_uint, img_as_float
# from google.cloud import storage
# from StringIO import StringIO
# from tensorflow.python.lib.io import file_io

img_dir = os.path.join(os.path.expanduser('~'), 'Desktop/MarsWorkSpace/Denoising_AutoEncoders/carvana/carvana/temp.npy')
temp_image=np.load(img_dir)
print(temp_image.shape)
print(temp_image.dtype)

####
temp_image = temp_image[:, :, 0]
# imgplot = plt.imshow(temp_image)
print(temp_image.shape)
matplotlib.image.imsave('sample_output.jpg', temp_image)
# im = exposure.rescale_intensity(temp_image, out_range='float')
# im = img_as_uint(im)
# io.imsave('test_16bit.png', im)
####
# print(temp_image)
print(".........")
# temp_image = temp_image * 255
# print("DEBUG: ", np.max(temp_image.flatten()))
# print("DEBUG: ", np.min(temp_image.flatten()))

# scipy.misc.imread(temp_image, flatten=False, mode='L')

# data = np.random.random((255,255))

#Rescale to 0-255 and convert to uint8
# rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

# im = Image.fromarray(rescaled)
# im.save('test.png')

# z=temp_image.astype(np.uint8)
# print(z)

# toimage(temp_image).show()
# w, h = 256, 256
# data = np.zeros((h, w, 1), dtype=np.uint8)
# data[256, 256] = [255, 0, 0]
# img = Image.fromarray(data, 'RGB')
# img.save('my.png')
# img.show()
# if temp_image.typecode() == Numeric.UnsignedInt8:
#         mode = "L"
# elif temp_image.typecode() == Numeric.Float32:
#         mode = "F"
# else:
#         print("unsupported image mode")
# x=Image.fromstring(mode, (temp_image.shape[1], temp_image.shape[0]), temp_image.tostring())
# print (x);