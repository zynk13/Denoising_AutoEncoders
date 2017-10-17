#!/usr/bin/python
import sys
from os import listdir
from os.path import isfile, join
from PIL import Image

GRAY_IMAGE_PREFIX="g_"
PATH_TO_SAVE_GRAY_IMAGES=str(sys.argv[1])+"gray/"
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
				print PATH_TO_SAVE_GRAY_IMAGES+GRAY_IMAGE_PREFIX+str(i)+".JPG"

				img = Image.open(fullfilename).convert('L')
				img=img.resize((256,256))
				img.save(PATH_TO_SAVE_GRAY_IMAGES+GRAY_IMAGE_PREFIX+str(i)+".png")
				i=i+1
			
if __name__ == '__main__':
  main()
#