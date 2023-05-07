from imutils import face_utils
import numpy as np
import argparse
import imutils
# dlib framework can be leveraged to train a shape predictor on the input training data — this is useful if you would like to train facial landmark detectors or custom shape predictors of your own.
import dlib
import cv2
from collections import OrderedDict
# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

FACIAL_LANDMARKS_IDXS = OrderedDict([

("mouth", (48,68)),
("right_eyebrow",(17,22)),
("left_eyebrow",(22,27)),
("right_eye",(36,42)),
("left_eye",(42,48)),
("nose",(27,35)),
("jaw",(0,17))
	])
def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()
	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]
	# loop over the facial landmark regions individually
	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
		# grab the (x, y)-coordinates associated with the
		# face landmark
		(j, k) = FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]
		# check if are supposed to draw the jawline
		if name == "jaw":
			# since the jawline is a non-enclosed facial region,
			# just draw lines between the (x, y)-coordinates
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)
		# otherwise, compute the convex hull of the facial
		# landmark coordinates points and display it
		else:
			# the Convex Hull of a shape or a group of points is a tight fitting convex boundary around the points or the shape.
			hull = cv2.convexHull(pts)
			# Contours are defined as the line joining all the points along the boundary of an image that are having the same intensity. 
			# Contours come handy in shape analysis, finding the size of the object of interest, and object detection.
			# Draw all contours
            # -1 signifies drawing all contours
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)
		# apply the transparent overlay
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	# return the output image
	return output
image = cv2.imread("OIP.jpg")
# image = imutils.resize(image , width=500)
# gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image 
# NOTE:Image bounding box dataset to detect faces in images.
#handles detecting the bounding box of faces in our image
# The second parameter is the number of image pyramid layers to apply when upscaling the image prior to applying the detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
rects = detector(image , 0)
shape=""
for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		# applies the facial landmark detector to the face region, returning a shape object which we convert to a NumPy array
		shape = predictor(image, rect)
		shape = face_utils.shape_to_np(shape)
image = cv2.imread("OIP.jpg")
res = visualize_facial_landmarks(image, shape, colors=None, alpha=0.75)
cv2.imshow("Output", res)
cv2.waitKey(0)