

import sys
import cv2
import numpy as np
import scipy.stats as st
import os
import math

def getimages():
    global filename1
    global filename2

    filename1 = input("Write a file name for the first image: \n")
    image1 = cv2.imread(filename1)

    while image1.shape[0] > 750 or image1.shape[1] > 1200:
        image1 = cv2.resize(image1,(int(image1.shape[1]/2), int(image1.shape[0]/2)))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = to3channel(image1)

    filename2 = input("Write a file name for the first image: \n")
    image2 = cv2.imread(filename2)

    while image2.shape[0] > 750 or image2.shape[1] > 1200:
        image2 = cv2.resize(image2,(int(image2.shape[1]/2), int(image2.shape[0]/2)))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = to3channel(image2)

    return image1, image2

def reloadimage():
    global filename1
    global filename2
    global image1
    global image2

    image1 = cv2.imread(filename1)

    while image1.shape[0] > 750 or image1.shape[1] > 1200:
        image1 = cv2.resize(image1,(int(image1.shape[1]/2), int(image1.shape[0]/2)))
    image1 = to3channel(image1)

    image2 = cv2.imread(filename2)

    while image2.shape[0] > 750 or image2.shape[1] > 1200:
        image2 = cv2.resize(image2,(int(image2.shape[1]/2), int(image2.shape[0]/2)))

    return image1, image2

def to3channel(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

#k slider
def sliderHandler(self):
    if self > 40 and self < 60:
        harris()
    return

#threhold slider
def sliderHandler2(self):
    if self != 0 and self < 100:
        harris()
    return

#winsize slider
def sliderHandler3(self):
    if self != 0:
        harris()
    return

def harris():
    img, img2 = reloadimage()
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k = cv2.getTrackbarPos('k', 'image1')/1000
    threshold = cv2.getTrackbarPos('threshold', 'image1')/10
    winsize = cv2.getTrackbarPos('winsize', 'image1')
    max_r = 0
    r = []
    M = np.matrix([[],[]])
    corner_list = []


    img1sx = img1
    img1sobelx = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=5)
    img1sx = cv2.normalize(img1sobelx, img1sx, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    img1sx = img1sx**2
    img1sy = img1
    img1sobely = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=5)
    img1sy = cv2.normalize(img1sobely, img1sy, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    img1sy = img1sy**2
    img1sxy = img1sx*img1sy


    for i in range(math.floor(winsize/2),(img1sx.shape[0])-math.floor(winsize/2)):
        for j in range(math.floor(winsize/2),(img1sx.shape[1])-math.floor(winsize/2)):
            Ix2 = img1sx[i-math.floor(winsize/2):i+math.floor(winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            Iy2 = img1sy[i-math.floor(winsize/2):i+math.floor(winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            IxIy = img1sxy[i-math.floor(winsize/2):i+math.floor(winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]

            Sx = Ix2.sum()
            Sy = Iy2.sum()
            Sxy = IxIy.sum()

            M = np.matrix([[Sx,Sxy],[Sxy,Sy]])
            det = np.linalg.det(M)
            tr = np.trace(M)
            r.append([i, j, det - k*tr**2])

    for pixel in r:
        if pixel[2] > max_r:
            max_r = pixel[2]

    for pixel in r:
        if pixel[2] >  threshold*max_r:
            corner_list.append((pixel[1], pixel[0]))

    img1 = to3channel(img1)

    while corner_list:
        corner = corner_list.pop()
        cv2.rectangle(img1, (corner[0]-5, corner[1]+5), (corner[0]+5, corner[1]-5), (0, 0, 255), 1)

    cv2.imshow('image1', img1)
    print("k = ", k)
    print("threshold = ", threshold)
    print("winsize = ", winsize)


    return

def features():
    img1, img2 = reloadimage()

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)
    print(bf)
    print(matches)
    cv2.imshow('features',img3)



def cornerHarris():
    img, img2 = reloadimage()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = to3channel(img)
    gray = np.float32(gray)
    cv2.imshow('example', img)
    dst = cv2.cornerHarris(gray,3,3,0.04)



    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('example',img)

def main():
    img1, img2 = getimages()

    cv2.imshow('example', img1)
    cv2.imshow('image1', img1)

    cv2.createTrackbar('k', 'image1', 40, 60, sliderHandler)
    cv2.createTrackbar('threshold', 'image1', 0, 10, sliderHandler2)
    cv2.createTrackbar('winsize', 'image1', 0, 10, sliderHandler3)






    while(True):
        key = cv2.waitKey()

        if key == ord('w'):
            cv2.imwrite("out.jpg", img)

        if key == ord('H'):
            cornerHarris()

        if key == ord('h'):
            harris()

        if key == ord('g'):
            features()

        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    main()
