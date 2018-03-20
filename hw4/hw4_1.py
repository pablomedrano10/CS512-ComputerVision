import cv2
import numpy as np
import glob


def main():

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    coords = np.zeros((6*7,3), np.float32)
    coords[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    points3D = []
    points2D = []

    images = glob.glob('*.jpg')

    for filename in images:
        image = cv2.imread(filename)
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(image_gray,(7,6),None)

        if ret == True:
            points3D.append(coords)

            corners_new = cv2.cornerSubPix(image_gray,corners,(11,11),(-1,-1),criteria)
            points2D.append(corners_new)

            img = cv2.drawChessboardCorners(image, (7,6), corners_new,ret)
            cv2.imshow('image',image)


    f = open("imagepoints.txt","w")
    for point in points2D:
        f.write("%s\n" %point)
    f.close()

    g = open("objectectpoints.txt","w")
    for point in points3D:
        g.write("%s\n" %point)
    g.close()


    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
