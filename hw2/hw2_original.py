#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:47:38 2017

@author: Pablo
"""

import cv2
import numpy as np
import sys
import math




#check if image name is specified in command line

def get_image():
    filename = input("Write a file name or press enter to capture an image: \n")
    if len(filename) > 1:
        original_image = cv2.imread(filename)
    else:
        cap = cv2.VideoCapture(0)
        retval,original_image = cap.read()
    while original_image.shape[0] > 1200 or original_image.shape[1] > 750:
        original_image = cv2.resize(original_image,(int(original_image.shape[1]/2), int(original_image.shape[0]/2)))
    cv2.imshow('image', original_image)
    return (original_image, filename)

def reloadimage(filename):
    if len(filename) > 1:
        original_image = cv2.imread(filename)
    else:
        cap = cv2.VideoCapture(0)
        retval,original_image = cap.read()
    while original_image.shape[0] > 1200 or original_image.shape[1] > 750:
        original_image = cv2.resize(original_image,(int(original_image.shape[1]/2), int(original_image.shape[0]/2)))
    return original_image


def sliderHandler(self):
    global image
    n = self
    image = reloadimage(filename)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if self != 0:
        kernel = np.ones((n,n), np.float32)/(n*n)
        image = cv2.filter2D(image, -1, kernel)
    cv2.imshow('image', image)

def convolve(image, kernel):
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) / 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")
    # loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
			k = (roi * kernel).sum()

			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k
    # rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	# return the output image
	return output

def sliderHandler2(self):
    global image
    n = self
    if self != 0:
        kernel = np.ones((n,n), np.float32)/(n*n)
        image = convolve(kernel, image)
    cv2.imshow('image', image)


def sliderHandler3(self):
    global image
    image = reloadimage(filename)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    new = image
    n = self
    if self != 0:
        kernel = np.ones((n,n), np.float32)/(n*n)
        sobelx = cv2.Sobel(new,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(new,cv2.CV_64F,0,1,ksize=5)
        for x in range(0, new.shape[0], n):
            for y in range(0, new.shape[1], n):
                grad_angle = math.atan2(sobely[x, y], sobelx[x, y])
                grad_x = int(x + n * math.cos(grad_angle))
                grad_y = int(y + n * math.sin(grad_angle))
                cv2.arrowedLine(new, (y, x), (grad_y, grad_x), (0, 0, 0))
    cv2.imshow('image', new)
    image = new

def sliderHandler4(self):
    global image
    image = reloadimage(filename)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    new = image
    n = self
    if self != 0:
        rot = cv2.getRotationMatrix2D((new.shape[1] / 2, new.shape[0] / 2), n, 1)
        new = cv2.warpAffine(new, rot,(new.shape[1], new.shape[0]))
    cv2.imshow('image', new)
    image = new

def main():
    global image
    global filename
    image, filename = get_image()


    count = 0

    while(True):
        key = cv2.waitKey()
        print (key)
        if key == ord('i'):
            image = reloadimage(filename)

        elif key == ord('w'):
            cv2.imwrite("out.jpg", image)

        elif key == ord('g'):
            image = reloadimage(filename)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 2:
                image_gray = image

        elif key == ord('G'):
            image = reloadimage(filename)
            if len(image.shape) == 3:
                aux = np.zeros((image.shape[0], image.shape[1]), dtype = image.dtype)
                for (x, y), v in np.ndenumerate(aux):
                    pixel = image[x, y, 0] * 0.299 + image[x, y, 1] * 0.587 + image[x ,y, 2] * 0.114
                    image[x, y] = pixel
            elif len(image.shape) == 2:
                image_gray = image

        elif key == ord('c'):
            image = reloadimage(filename)
            if len(image.shape) == 3:
                if count == 0:
                    image[:,:,1] = 0
                    image[:,:,2] = 0
                    count = 1
                elif count == 1:
                    image[:,:,0] = 0
                    image[:,:,2] = 0
                    count = 2
                else:
                    image[:,:,0] = 0
                    image[:,:,1] = 0
                    count = 0
            else:
                print("Not possible to convert to b, g or r")

        elif key == ord('s'):
           image = reloadimage(filename)
           image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
           cv2.imshow('image', image)
           cv2.createTrackbar('s', 'image', 0, 255, sliderHandler)

        elif key == ord('S'):
            image = reloadimage(filename)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            cv2.imshow('image', image)
            cv2.createTrackbar('s', 'image', 0, 255, sliderHandler2)

#
#
        elif key == ord('d'):
            image = reloadimage(filename)
            image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
#
#
        elif key == ord('D'):
            image = reloadimage(filename)
            image = cv2.pyrDown(image)

        elif key == ord('x'):
            image = reloadimage(filename)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 2:
                image_gray = image
            sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
            image = cv2.normalize(sobelx, image, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        elif key == ord('y'):
            image = reloadimage(filename)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 2:
                image_gray = image
            sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
            image = cv2.normalize(sobely, image, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
#
#
        elif key == ord('m'):
            image = reloadimage(filename)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 2:
                image_gray = image
            sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
            sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
            gradient = cv2.magnitude(sobelx, sobely)
            image = cv2.normalize(gradient, image, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        elif key == ord('p'):
            image = reloadimage(filename)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 2:
                image_gray = image
            cv2.createTrackbar('s', 'image', 0,255, sliderHandler3)


        elif key == ord('r'):
            image = reloadimage(filename)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 2:
                image_gray = image
            cv2.createTrackbar('s', 'image', 0, 360, sliderHandler4)


        elif key == 27:
            cv2.destroyAllWindows()
            break

        cv2.imshow('image', image)



if __name__ == '__main__':
    main()
