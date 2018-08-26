
import cv2 as cv #import OpenCV

#This method is used to show images without being enlarged. Try only with `cv.imshow` to see the difference
def showimage(img):
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', window_width, window_height)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

c=0 #counter for number of images in the collage

im = cv.imread('collage5.jpg') #input the image
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY) #convert it to Gray Scale image
ret, thresh = cv.threshold(imgray, 250, 255, 0) #convert it into a binary image(0 or 255). We only care about the threshold image.
im2, conts, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #finding continous blobs of pixels. We only care about conts.
#cv.RETR_TREE will find continous blobs of pixels within the picture. Try with cv.RETR_EXTERNAL to see the difference
#cv.CHAIN_APPROX_SIMPLE will give only few and extreme coordinates of contours. cv.CHAIN_APPROX_NONE will detect all the continous blobs of pixels
cv.drawContours(im, conts, -1, (0,255,0), 3) #drawing those contours on the image.

q=3 #this is to eliminate the contours line from the cropped images

for cnt in conts:
    area = cv.contourArea(cnt) #area of contours detected.

    if area > 20: #eliminate small contours
        c+=1
        x1, y1, w, h = cv.boundingRect(cnt) #finding rectangles
        x2 = x1 + w
        y2 = y1 + h
        print("x1:", x1, " y1:", y1, " x2:", x2, " y2:", y2) #(x1, y1): coordinates of rectangles for 1st vertex. (x2, y2): coordinates of rectangles for diagonally opposite vertex.
        crop_img = im[y1+q:y1+h-q, x1+q:x1+w-q] #remove q from this line to understand what it really does
        cv.imwrite("cropped"+str(c)+'.jpg', crop_img) #saving the cropped images
showimage(im)
