import cv2
import glob




def selectObject(image):
    image = image
    x1 = None
    y1 = None
    x2 = None
    y2 = None

    def drawSquare(event, x, y, flags, param):
        # global points
        # global points, image

        if event == cv2.EVENT_LBUTTONDOWN:
            x1 = x
            y1 = y

        elif event == cv2.EVENT_LBUTTONUP:
            x2 = x
            y2 = y
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imshow('image', image)



    cv2.setMouseCallback("image", drawSquare)

    while True:
        # display the image and wait for a keypress
        cv2.imshow('image', image)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("c"): # Hit 'c' to confirm the selection
            break

    # close the open windows
    # cv2.destroyAllWindows()

    return x1, y1, x2, y2



def main():
    path = input("Write the positive images' absolute path: \n")
    f = open('positives2.txt', 'w')

    for filename in glob.glob(path+'/*.jpg'):
        image = cv2.imread(filename)
        cv2.imshow('image', image)

        x1, y1, x2, y2 = selectObject(image)
        print(x1, y1, x2, y2)
        width = x2-x1
        height = y2-y1



        while (True):
            key = cv2.waitKey()

            if key == ord('s'):
                f.write(filename+' '+'1 '+str(x1)+' '+str(y1)+' '+str(width)+' '+str(height))
                cv2.destroyAllWindows()
                break

            elif key == 27:
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    main()
