This is a program which can output all the images in a collage, when an collage with white borders is inputted.

The cropped pictures are output images when Collage1 is fed as input.

The collage_border_detector.py uses findContours method from OpenCv to detect borders of the images. Works pretty good with large and white borders.

The test.py is an attempt to detect rectangles without using OpenCV. This can only work with rectangular pictures even with little border. But the code doesn't work properly. If you can improve it, please send me a PR. 

Most of the code is docummented.
