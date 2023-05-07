# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
# dlib framework can be leveraged to train a shape predictor on the input training data — this is useful if you would like to train facial landmark detectors or custom shape predictors of your own.
import dlib
import cv2
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())
#below , which is assumed to be a bounding box rectangle produced by a dlib detector (i.e., the face detector).
def rectangle_to_bounding_box(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x=rect.left()
	y=rect.top()
	w = rect.right() - x
	h = rect.bottom()  - y
	return (x , y , w , h)
# The dlib face landmark detector will return a shape object containing the 68 (x, y)-coordinates of the facial landmark regions.
def shape_to_numpy_array(shape,dtype="int"):
	# initialize the list of (x,y)-coordinates
	coordinates = np.zeros((68,2),dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0,68):
		coordinates[i] = (shape.part(i).x , shape.part(i).y)
# return the list of (x, y)-coordinates
	return coordinates
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
# This is the path to dlib’s pre-trained facial landmark detector. 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# load the input image, resize it, and convert it to grayscale
image = cv2.imread("th.jpg")
image = imutils.resize(image , width=500)
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image 
# NOTE:Image bounding box dataset to detect faces in images.
#handles detecting the bounding box of faces in our image
# The second parameter is the number of image pyramid layers to apply when upscaling the image prior to applying the detector
rects = detector(gray , 1)
# cv2.imshow("Output", gray)
# cv2.waitKey(0)
# print(rects)
# loop over the face detections
for (i,rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray,rect)
	shape = face_utils.shape_to_np(shape)
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x ,y , w , h) = face_utils.rect_to_bb(rect)
	# method is used to draw a rectangle on any image
	#cv2.rectangle(image, start_point, end_point, color, thickness)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	# cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		# cv2.circle(image, center_coordinates, radius, color, thickness)
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)





