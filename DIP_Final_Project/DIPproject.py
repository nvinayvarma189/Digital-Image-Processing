import threading
from cv2.cv2 import HOGDescriptor
from skimage import img_as_float
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy
import cv2
import scipy.spatial.distance
import time


class Features:
    def __init__(self, query_keys: list, query_des: numpy.ndarray, train_keys: numpy.ndarray, train_des: numpy.ndarray):
        """this is the constuctor of the method"""
        self.query_keys = query_keys
        self.query_des = query_des
        self.train_keys = train_keys
        self.train_des = train_des


class Channels:
    def __init__(self, red: int, green: int, blue: int):
        """this will help to hold the average value of three channel pixels in a segment"""
        self.red = red
        self.green = green
        self.blue = blue


def getImageSegments(rgbImage: numpy.uint8, segments: int, sigma: int) -> numpy.ndarray:
    """this will return the slic segmetns"""
    # img = img_as_float(image)
    bgrImage = rgbImage[:, :, ::-1]  # converted to bgr order because skimage work on bgr images
    segmentedData = slic(image=bgrImage, n_segments=segments, sigma=sigma)
    drawSegments(image=bgrImage, segments=segmentedData)
    return segmentedData

def drawSegments(image: numpy.float, segments: numpy.ndarray) -> None:
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (50))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")
    # show the plots
    plt.show()
    return None


def getMostAppropriteSegementNumber(image: numpy.ndarray) -> int:
    """shape is in the order ROWS,COLS,CHANNELS -> (y,x,c)"""
    y, x, c = image.shape
    totalSegments = int(round((x * y) / (50 * 50)))
    print('Total Segments', totalSegments)
    return totalSegments
    # return 50



def resizeImage(image: numpy.uint8) -> numpy.uint8:
    """this will resize image if dimensions are greater than 512"""
    if image.shape[0] > 1100:
        width = int(numpy.around((image.shape[1]) / 2))
        height = int(numpy.around(image.shape[0] / 2))
        print('height , width:',width,',',height)
        resize_image = cv2.resize(src=image, dsize=(width, height))
        return resizeImage(resize_image)
    return image


def getLocalizedImageSegment(segmentValue: int, segments: numpy.ndarray, image: numpy.ndarray) -> (
        numpy.ndarray, numpy.ndarray):
    rows, cols = numpy.where(segments == segmentValue)

    '''image ROI is taken as image[y1:y2,x1:x2]'''
    roi = image[min(rows):max(rows), min(cols):max(cols)]

    '''removing the roi from the full image'''
    imageWithRemovedROI = image.copy()
    imageWithRemovedROI[min(rows):max(rows), min(cols):max(cols)] = 0

    return roi, imageWithRemovedROI


def getArrangedDictionary(segments, image):
    """this will prepare a dictionary of key points and their descriptors of each approximated patches"""
    sift = cv2.xfeatures2d.SIFT_create()

    dictionary = {}

    '''get the unique values of the segmentation mask'''
    unique_segment_values = numpy.unique(segments)
    '''iterate over the segment to get the localized sift keypoints'''
    for segmentValue in unique_segment_values:
        local_image_segment, image_without_roi = getLocalizedImageSegment(segmentValue=segmentValue, segments=segments,
                                                                          image=image)

        query_keys, query_des = sift.detectAndCompute(cv2.cvtColor(local_image_segment,cv2.COLOR_BGR2GRAY), None)
        train_keys, train_des = sift.detectAndCompute(cv2.cvtColor(image_without_roi,cv2.COLOR_BGR2GRAY), None)

        '''adding only if at least one key point is found'''
        if query_des is not None:
            dictionary[segmentValue] = Features(query_keys=query_keys, query_des=query_des, train_keys=train_keys,
                                                train_des=train_des)

    return dictionary


def getMatchedPatches(arranged_dictionary: dict, segments: numpy.ndarray, key_matches_per_cluster=5,
                      cluster_matches_per_cluster=2) -> dict:

    bf = cv2.BFMatcher()  # BFMatcher with default params

    matched_segments = {}  # the dictionary to hold the matched segments
    for segmentValue in arranged_dictionary:
        dic_value = arranged_dictionary[segmentValue]
        '''to make sure that we check the segment combinations that have not met before'''
        condition = dic_value.query_des is not None
        if condition:
            matches = bf.knnMatch(queryDescriptors=dic_value.query_des, trainDescriptors=dic_value.train_des, k=2)
            count = 0  # to keep the count of the matches under the threshold
            matching_segments = set()
            '''iterating over the matches to filter out those under the required threshold'''
            for i, m in enumerate(matches):
                if len(m) > 1:
                    if m[0].distance < 0.4 * m[1].distance:
                    	'''now we have found a good match'''
                    	count += 1
                    	'''get the segment value of the matched key point'''
                    	col, row = dic_value.train_keys[m[0].trainIdx].pt  # key.pt -> (x,y)|(col,row)|(width,height)
                    	matching_segments.add(segments[int(round(row)), int(round(col))])

            '''filling the segment into the dictionary if there are required number of matches in the patches'''
            if (count >= key_matches_per_cluster) & (len(matching_segments) >= cluster_matches_per_cluster):
                print('matching...', segmentValue, ',', matching_segments)
                if segmentValue in matched_segments:
                    matched_segments[segmentValue].extend(
                        x for x in matching_segments if x not in matched_segments[segmentValue])
                else:
                    matched_segments[segmentValue] = matching_segments

        else:
            continue

    print(matched_segments, "ms")
    return matched_segments


def drawMatchedClusters(image: numpy.ndarray, matchedClusters: dict, segments: numpy.ndarray) -> None:
    """this will draw the matched clusters"""
    for matchedClusterNumber in matchedClusters:
        print('matched parent cluster:', matchedClusterNumber, ' |children :', end='', flush=True)

        rows1, cols1 = numpy.where(segments == matchedClusterNumber)
        for row1, col1 in zip(rows1, cols1):
            image[row1, col1, 1] = 0  # (255, 255, 255)

        if len(matchedClusters[matchedClusterNumber]) >= 1:
            for similarClusterNumber in matchedClusters[matchedClusterNumber]:
                rows2, cols2 = numpy.where(segments == similarClusterNumber)
                print(similarClusterNumber, ',', end='')

                ''' Draw a diagonal blue line with thickness of 5 px parameters: pt1 is in (x,y) order '''
                cv2.line(img=image, pt1=(cols1[0], rows1[0]), pt2=(cols2[0], rows2[0]), color=(255, 0, 0), thickness=2)

                for row2, col2 in zip(rows2, cols2):
                    image[row2, col2, 2] = 0  # (255, 255, 255)
        print('\n')

    cv2.imshow('clone detected image', (image))





start_time = time.time()
img = resizeImage(cv2.imread('/home/roronoa/Desktop/Projects/DIP/project/experiments/copyLenna.png'))

# ''' using SIFT descriptor with brute force KNN match
segs = getImageSegments(rgbImage=img, segments=getMostAppropriteSegementNumber(image=img), sigma=5)
print('unique segments :', len(numpy.unique(segs)))
arrangedDict = getArrangedDictionary(segments=segs, image=img)
print('arranged dictionary', len(arrangedDict))
matchedPatches = getMatchedPatches(arranged_dictionary=arrangedDict, segments=segs, key_matches_per_cluster=2,
                                   cluster_matches_per_cluster=2)
print('matched patches:', len(matchedPatches))
drawMatchedClusters(image=img, matchedClusters=matchedPatches, segments=segs)

print('time of execution - ', time.time() - start_time)
cv2.waitKey(0)
cv2.destroyAllWindows()
