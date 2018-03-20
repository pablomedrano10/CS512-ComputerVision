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
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = to3channel(image1)

    image2 = cv2.imread(filename2)

    while image2.shape[0] > 750 or image2.shape[1] > 1200:
        image2 = cv2.resize(image2,(int(image2.shape[1]/2), int(image2.shape[0]/2)))

    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = to3channel(image2)

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
    image1, image2 = reloadimage()

    image1_g = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_g = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


    k = cv2.getTrackbarPos('k', 'image')/1000
    threshold = cv2.getTrackbarPos('threshold', 'image')/10
    winsize = cv2.getTrackbarPos('winsize', 'image')

    max_r1 = 0
    r1 = []
    M1 = np.matrix([[],[]])
    corner_list1 = []

    max_r2 = 0
    r2 = []
    M2 = np.matrix([[],[]])
    corner_list2 = []

    #calculate conerners for image1
    image1sx = image1_g
    image1sobelx = cv2.Sobel(image1_g,cv2.CV_64F,1,0,ksize=5)
    image1sx = cv2.normalize(image1sobelx, image1sx, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    image1sx = image1sx**2
    image1sy = image1_g
    image1sobely = cv2.Sobel(image1_g,cv2.CV_64F,0,1,ksize=5)
    image1sy = cv2.normalize(image1sobely, image1sy, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    image1sy = image1sy**2
    image1sxy = image1sx*image1sy


    for i in range(math.floor(winsize/2),(image1sx.shape[0])-math.floor(winsize/2)):
        for j in range(math.floor(winsize/2),(image1sx.shape[1])-math.floor(winsize/2)):
            Ix2_1 = image1sx[i-math.floor(winsize/2):i+math.floor(winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            Iy2_1 = image1sy[i-math.floor(winsize/2):i+math.floor(winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            IxIy_1 = image1sxy[i-math.floor(winsize/2):i+math.floor(winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]

            Sx_1 = Ix2_1.sum()
            Sy_1 = Iy2_1.sum()
            Sxy_1 = IxIy_1.sum()

            M1 = np.matrix([[Sx_1,Sxy_1],[Sxy_1,Sy_1]])
            det1 = np.linalg.det(M1)
            tr1 = np.trace(M1)
            r1.append([i, j, det1 - k*tr1**2])

    for pixel in r1:
        if pixel[2] > max_r1:
            max_r1 = pixel[2]

    for pixel in r1:
        if pixel[2] >  threshold*max_r1:
            corner_list1.append((pixel[1], pixel[0]))

    #calculate conerners for image2
    image2sx = image2_g
    image2sobelx = cv2.Sobel(image2_g,cv2.CV_64F,1,0,ksize=5)
    image2sx = cv2.normalize(image2sobelx, image2sx, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    image2sx = image2sx**2
    image2sy = image2_g
    image2sobely = cv2.Sobel(image2_g,cv2.CV_64F,0,1,ksize=5)
    image2sy = cv2.normalize(image2sobely, image2sy, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    image2sy = image2sy**2
    image2sxy = image2sx*image2sy


    for i in range(math.floor(winsize/2),(image2sx.shape[0])-math.floor(winsize/2)):
        for j in range(math.floor(winsize/2),(image2sx.shape[1])-math.floor(winsize/2)):
            Ix2_2 = image2sx[i-math.floor(winsize/2):i+math.floor(winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            Iy2_2 = image2sy[i-math.floor(winsize/2):i+math.floor(winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            IxIy_2 = image2sxy[i-math.floor(winsize/2):i+math.floor(winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]

            Sx_2 = Ix2_2.sum()
            Sy_2 = Iy2_2.sum()
            Sxy_2 = IxIy_2.sum()

            M2 = np.matrix([[Sx_2,Sxy_2],[Sxy_2,Sy_2]])
            det2 = np.linalg.det(M2)
            tr2 = np.trace(M2)
            r2.append([i, j, det2 - k*tr2**2])

    for pixel in r2:
        if pixel[2] > max_r2:
            max_r2 = pixel[2]

    for pixel in r2:
        if pixel[2] >  threshold*max_r2:
            corner_list2.append((pixel[1], pixel[0]))

    #calculate features
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(image1,None)
    kp2, des2 = orb.detectAndCompute(image2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    image = cv2.drawMatches(image1,kp1,image2,kp2,matches[:20],None, flags=2)
    cv2.imshow('image',image)

    #draw corners
    while corner_list1:
        corner1 = corner_list1.pop()
        cv2.rectangle(image, (corner1[0]-5, corner1[1]+5), (corner1[0]+5, corner1[1]-5), (0, 0, 255), 1)

    while corner_list2:
        corner2 = corner_list2.pop()
        cv2.rectangle(image, (image1.shape[1]+corner2[0]-5, corner2[1]+5), (image1.shape[1]+corner2[0]+5, corner2[1]-5), (0, 0, 255), 1)

    cv2.imshow('image', image)
    print("k = ", k)
    print("threshold = ", threshold)
    print("winsize = ", winsize)


    return


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
    image1, image2 = getimages()

    print("press h to get help")

    image = np.concatenate((image1, image2), axis = 1)
    cv2.imshow('example', image1)
    cv2.imshow('image', image)

    cv2.createTrackbar('k', 'image', 40, 60, sliderHandler)
    cv2.createTrackbar('threshold', 'image', 0, 10, sliderHandler2)
    cv2.createTrackbar('winsize', 'image', 0, 10, sliderHandler3)






    while(True):
        key = cv2.waitKey()

        if key == ord('C'):
            cornerHarris()

        elif key == ord('c'):
            harris()

        elif key == ord('h'):
            print("press C to find corners using Harris corner detection openCV function")
            print("press c to find corners using my algorithm and match feature points")

        elif cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    main()
