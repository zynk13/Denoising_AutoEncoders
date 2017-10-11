import numpy
from PIL import Image
import os
import pylab as pl
import numpy as np
import glob

img_dir = os.path.join(os.path.expanduser('~'), 'Desktop/MarsWorkSpace/Denoising_AutoEncoders/carvana/test')
output_dir = os.path.join(os.path.expanduser('~'), 'Desktop/MarsWorkSpace/Denoising_AutoEncoders/carvana/output/output_original_resized/')
total_fns = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(img_dir, '*.jpg'))]


i=0
for image in total_fns:
	print(img_dir+"/"+image+".jpg")
	img_path=img_dir+"/"+image+".jpg"
	img = Image.open(img_path)
	img=img.resize((256,256))
	img.save(output_dir+image+".jpg")