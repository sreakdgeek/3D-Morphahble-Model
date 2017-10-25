# This module is built based on dlib implementation and face land mark detection code from learnopencv.com

import cv2 
import dlib 
import numpy 
import numpy as np
import sys
from matplotlib import pyplot as plt
import glob
from scipy import sparse, io

PREDICTOR_PATH = "/home/pyimagesearch/fab_vision/shape_extraction/shape_predictor_68_face_landmarks.dat"

WHITE_COLOR = [255, 255, 255]
SCALE_FACTOR = 1  
FEATHER_AMOUNT = 11 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise ValueError("The input image has more than 1 Faces!")
    if len(rects) == 0:
        raise ValueError("The input image has NO Face!")

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

def DisplayImages(original, annotateImg, extractedFaceShape):
    plt.axis("off")
    mergedImage = numpy.concatenate((original, annotateImg, extractedFaceShape), axis=1)
    print extractedFaceShape.shape

    #face_skin = sparse.csr_matrix(extractedFaceShape)
    #io.mmwrite("face_skin.mtx", face_skin)
    np.save("output/face_skin.npy", extractedFaceShape)

    plt.imshow(cv2.cvtColor(mergedImage, cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(extractedFaceShape, cv2.COLOR_BGR2RGB))
    plt.show()

def ExtractFaceShapesFromImages(folderPath):
    for filename in glob.iglob(folderPath):
        try:
            img, landmarks = read_im_and_landmarks(filename)
            annotateImg = annotate_landmarks(img,landmarks)
            blank_image = numpy.zeros(img.shape, numpy.uint8)
            points = cv2.convexHull(landmarks)

            cv2.fillConvexPoly(blank_image, points, color=WHITE_COLOR)
            extractedFaceShape = cv2.bitwise_and(blank_image,img)
            DisplayImages(img, annotateImg, extractedFaceShape)    
        except ValueError as err:
            print "Error Processing Image : ", filename
            print err
            plt.axis("off")
            plt.imshow(cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            plt.show()
            continue

ExtractFaceShapesFromImages('/home/pyimagesearch/fab_vision/shape_extraction/images/*.jpg')
