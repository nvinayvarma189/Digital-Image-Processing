# import numpy as np
import cv2 as cv

def showimage(img):
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    # print("working")
    cv.namedWindow('gray', cv.WINDOW_NORMAL)
    cv.resizeWindow('gray', window_width, window_height)
    cv.imshow('gray', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

c=0
im = cv.imread('collage5.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 250, 255, 0)
im2, conts, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(im, conts, -1, (0,255,0), 3)
q=3
for cnt in conts:
    area = cv.contourArea(cnt)

    if area > 20:
        c+=1
        x1, y1, w, h = cv.boundingRect(cnt)
        x2 = x1 + w
        y2 = y1 + h
        print("x1:", x1, " y1:", y1, " x2:", x2, " y2:", y2)
        crop_img = im[y1+q:y1+h-q, x1+q:x1+w-q]
        cv.imwrite("cropped"+str(c)+'.jpg', crop_img)
showimage(im)
