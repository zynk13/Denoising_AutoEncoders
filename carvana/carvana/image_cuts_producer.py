import numpy
import collections
import matplotlib.pyplot as plt
import io
import numpy as np 
from PIL import Image
import os
import matplotlib
import matplotlib.image as mpimg

def image_cuts_producer():
	img_dir = os.path.join(os.path.expanduser('~'), 'Desktop/MarsWorkSpace/Denoising_AutoEncoders/carvana/carvana/image_cuts.npy')
	output_dir = os.path.join(os.path.expanduser('~'), 'Desktop/MarsWorkSpace/Denoising_AutoEncoders/carvana/output/output_cuts/')
	image_cuts_array=np.load(img_dir)

	i=0
	for image in image_cuts_array:
		print(image.shape)
		print(image.dtype)
		image = image[:, :, 0]
		matplotlib.image.imsave(output_dir+str(i)+'.jpg', image)
		i+=1