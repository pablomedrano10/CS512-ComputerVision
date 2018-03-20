import cv2
import numpy as np
import math

def getimage():
    global filename
    filename = input("Write a file name or press enter to capture an image: \n")
    original_image = cv2.imread(filename)

    while original_image.shape[0] > 1200 or original_image.shape[1] > 750:
        original_image = cv2.resize(original_image,(int(original_image.shape[1]/2), int(original_image.shape[0]/2)))
    return original_image

def reloadimage():
    global filename
    original_image = cv2.imread(filename)

    while original_image.shape[0] > 1200 or original_image.shape[1] > 750:
        original_image = cv2.resize(original_image,(int(original_image.shape[1]/2), int(original_image.shape[0]/2)))
    return original_image

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
    img = reloadimage()
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k = cv2.getTrackbarPos('k', 'image1')/1000
    threshold = cv2.getTrackbarPos('threshold', 'image1')/10
    winsize = cv2.getTrackbarPos('winsize', 'image1')
    max_r = 0
    r = []

    corner_list = []


    m = cv2.cornerEigenValsAndVecs(np.array(img1, dtype="float32"), 3, 3)
    print(m.shape)
    for i in range(0, m.shape[0]):
        for j in range(0, m.shape[1]):
            r.append([i, j, m[0]*m[1]-k*((m[0]+m[1])**2)])

    print(r)

    for pixel in r:
        if pixel[2] > threshold*max_r:
            max_r = pixel[2]

    for pixel in r:
        if pixel[2] >  threshold*max_r:
            corner_list.append((pixel[1], pixel[0]))

    while corner_list:
        x = corner_list.pop()
        cv2.rectangle(img,(x[0]-1,x[1]+1),(x[0]+1,x[1]-1),(0,0,255),1)

    cv2.imshow('image1', img)
    print("k = ", k)
    print("threshold = ", threshold)
    print("winsize = ", winsize)


    return

def cornerHarris():
    img = reloadimage()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    cv2.imshow('dst', img)
    dst = cv2.cornerHarris(gray,3,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst',img)

def main():
    img = getimage()

    cv2.imshow('dst', img)
    cv2.imshow('image1', img)
    cv2.createTrackbar('k', 'image1', 40, 60, sliderHandler)
    cv2.createTrackbar('threshold', 'image1', 0, 10, sliderHandler2)
    cv2.createTrackbar('winsize', 'image1', 0, 10, sliderHandler3)





    while(True):
        key = cv2.waitKey()

        if key == ord('H'):
            cornerHarris()

        if key == ord('h'):
            harris()


        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    main()
