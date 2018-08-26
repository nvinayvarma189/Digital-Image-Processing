## Description

1.This is a program which can output all the images in a collage, when an collage with white borders is inputted.

2.The cropped pictures are output images when `Collage1.jpg` is fed as input.

## files

The `collage_border_detector.py` uses `findContours` method from `OpenCv` to detect borders of the images. Works pretty good with large and white borders.

The `test.py` is an attempt to detect rectangles without using OpenCV. This can only work with rectangular pictures even with little border. But the code doesn't work properly. Will improve soon.

### Thank you

Most of the code is documented. If you can improve anything, please send me a PR. 
