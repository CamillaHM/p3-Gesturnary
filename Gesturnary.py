import sys

import numpy as np
import cv2
from contextlib import suppress

# Toggle between using camera or test video
Realcam = False

# True, to use camera as input
if Realcam is True:
    cap = cv2.VideoCapture(0)

# False, to use video as input
if Realcam is False:
    cap = cv2.VideoCapture('Vid1.mp4')


while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Blur image
        blurred_frame = cv2.bilateralFilter(frame, 9, 75, 75)

        # Convert from BGR to HSV
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # Defining the range of color in HSV
        lower_hand = np.array([40, 100, 80])
        upper_hand = np.array([80, 255, 255])

        # Threshold the HSV image to only get hand colors
        mask = cv2.inRange(hsv, lower_hand, upper_hand)

        # Contours

        with suppress(Exception):
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        with suppress(Exception):
            image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, contours, -2, (0, 255, 0), 3)

        if contours:
            # assume largest contour is the one of interest
            max_contour = max(contours, key=cv2.contourArea)

            epsilon = 0.01 * cv2.arcLength(max_contour, True)
            max_contour = cv2.approxPolyDP(max_contour, epsilon, True)

            # determine convex hull & convexity defects of the hull
            hull = cv2.convexHull(max_contour, returnPoints=False)
            defects = cv2.convexityDefects(max_contour, hull)

            # Camera

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])
                    cv2.line(frame, start, end, [255, 255, 0], 2)
                    cv2.circle(frame, far, 5, [0, 0, 255], -1)

                    # convert image to grayscale image
                    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # convert the grayscale image to binary image
                    ret, thresh = cv2.threshold(gray_image, 40, 100, 80)

                    # calculate moments of binary image
                    M = cv2.moments(thresh)

                    # calculate x,y coordinate of center
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # put text and highlight the center
                    cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
                    cv2.putText(frame, "Center of glove", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

                    # Display
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

    else:
        # replay
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()