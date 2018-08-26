from __future__ import division
import numpy as np
import cv2

img = cv2.imread('collage4.jpg')

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
ret,thresh_img = cv2.threshold(img,252,255,cv2.THRESH_BINARY)
cv2.imwrite('threshold.jpg', thresh_img)
c = 0

b1=l1=b2=l2=-1
for x in range(thresh_img.shape[0]):
    for y in range(thresh_img.shape[1]):
        # print(x,y)
        if(thresh_img[x][y] == 0 and (x<b1 or x>b2) and (y<l1 or y>l2)):
            c+=1
            print(x,y)
            a=x
            b=y
            # print(thresh_img[a][b])
            b1 = a
            while(thresh_img[a+1][b] != 255):
                a=a+1
            b2 = a
            l1 = b
            while(thresh_img[a][b+1] != 255):
                b+=1
            l2 = b
            print(b1,b2,l1,l2)
            single_image = img[b1:b2, l1:l2]
            # showimage(single_image)
            cv2.imwrite('pic'+str(c)+'.png', single_image)

def showimage(img):
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    # print(1)
    cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('gray', window_width, window_height)
    cv2.imshow('gray', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# print(thresh_img[100][100])
''' height  =  row
    widht   =  column
'''
