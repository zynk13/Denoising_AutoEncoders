#!/usr/bin/python
import sys
from os import listdir
from os.path import isfile, join
from PIL import Image

FENCED_IMAGE_PREFIX="f_"
PATH_TO_SAVE_GRAY_IMAGES=str(sys.argv[1])+"fenced/"
FENCE="/Users/mohitakhakharia/Desktop/MarsWorkSpace/Denoising_AutoEncoders/fence/g_fence.png"
def main():
	print 'Path in the argument:', str(sys.argv[1])
	dirlist=listdir(str(sys.argv[1]))
	
	i=1
	for x in dirlist:	
		if(isfile(join(str(sys.argv[1]), x))):
			if x!=".DS_Store":
				fullfilename= str(sys.argv[1])+x
				print "Printing current file x-"
				print x
				
				print "Saving converted file -"
				print PATH_TO_SAVE_GRAY_IMAGES+FENCED_IMAGE_PREFIX+str(i)+".png"

				img = Image.open(fullfilename);
				fence_img = Image.open(FENCE);
				fence_img = fence_img.convert('L')
				# img = Image.blend(img, fence_img, 0.9)
				img.paste(fence_img, (0, 0), fence_img)
				img.save(PATH_TO_SAVE_GRAY_IMAGES+FENCED_IMAGE_PREFIX+str(i)+".png")
				i=i+1
				
			
if __name__ == '__main__':
  main()
#