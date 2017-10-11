import numpy
from PIL import Image
import os
import pylab as pl
import numpy as np
import glob

output_dir = os.path.join(os.path.expanduser('~'), 'Desktop/MarsWorkSpace/Denoising_AutoEncoders/carvana/output/')
img_output_cuts_dir = os.path.join(os.path.expanduser('~'), 'Desktop/MarsWorkSpace/Denoising_AutoEncoders/carvana/output/output_cuts/')
img_output_original_resized_dir = os.path.join(os.path.expanduser('~'), 'Desktop/MarsWorkSpace/Denoising_AutoEncoders/carvana/output/output_original_resized/')

img_output_cuts_dir_fns = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(img_output_cuts_dir, '*.jpg'))]
img_output_original_resized_fns = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(img_output_original_resized_dir, '*.jpg'))]

for i in range(len(img_output_cuts_dir_fns)):
	print(i)
	
	print(img_output_cuts_dir+img_output_cuts_dir_fns[i]+".jpg")
	image_cut=img_output_cuts_dir+img_output_cuts_dir_fns[i]+".jpg"

	print(img_output_original_resized_dir+img_output_original_resized_fns[i]+".jpg")
	image_original=img_output_original_resized_dir+img_output_original_resized_fns[i]+".jpg"
		
	background = Image.open(image_original)
	overlay = Image.open(image_cut)

	background = background.convert("RGBA")
	overlay = overlay.convert("RGBA")

	new_img = Image.blend(background, overlay, 0.5)
	new_img.save(output_dir+img_output_original_resized_fns[i]+".png","PNG")