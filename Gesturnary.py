import numpy as np
import cv2
from contextlib import suppress
import math

# Settings # Settings # Settings
import matplotlib.pyplot as plt


# Use real camera or test videos
Realcam = True

# Which test video to use
Video = 3
# 1 - One finger
# 2 - Open hand
# 3 - Switches between open and closed hand

# PenMarker settings
pMarker = True
pMarkerSize = 20
pMarkerPos = []
pMarkerColor = (255, 0, 0)
pMarkerThick = -1

# EraserMarker settings
eMarker = True
eMarkerSize = 20
eMarkerColor = (0, 0, 0)
eMarkerThick = 1

# Pen settings
draw = True
PenSize = 4
PenColor = (0, 0, 0)

# Eraser settings
eraser = False
EraserSize = 20
EraserColor = (255, 255, 255)

# End of settings # End of settings # End of settings
MinCountourSize = 3000
Playonce = False
lastend = ()
Max_Fingers = 4
FingerVid = "Vid1.mp4"
OpenVid = "Vid2.mp4"
OpenAndClosedVid = "Vid3.mp4"

if Realcam:
    cap = cv2.VideoCapture(0)

if not Realcam:
    if Video == 1:
        cap = cv2.VideoCapture(FingerVid)
    if Video == 2:
        cap = cv2.VideoCapture(OpenVid)
    if Video == 3:
        cap = cv2.VideoCapture(OpenAndClosedVid)

WhiteBK = False
end=()
cnt = 0
drawing = ()
center=()
MarkerFrame = ()
key = cv2.waitKey(1)


def calculateFingers(res, drawing):  # -> finished bool, cnt: finger count
    #  convexity defect

    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem

                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    if cnt <= Max_Fingers:  # stop finding fingers after x number has been found
                        cnt += 1
                        cv2.circle(frame, far, 8, [211, 84, 0], -1)

            return True, cnt


while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # C to clear drawing
    if key & 0xFF == ord('c'):
        drawing = np.full(frame.shape, 255, dtype=np.uint8)

        WhiteBK = False
        print("cleared drawing")

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

        # if either fail, they will suppress and try the other
        with suppress(Exception):
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        with suppress(Exception):
            image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:

            key = cv2.waitKey(1)

            # assume largest contour is the one of interest
            max_contour = max(contours, key=cv2.contourArea)

            epsilon = 0.01 * cv2.arcLength(max_contour, True)
            max_contour = cv2.approxPolyDP(max_contour, epsilon, True)

            # determine convex hull & convexity defects of the hull
            hull = cv2.convexHull(max_contour, returnPoints=False)
            defects = cv2.convexityDefects(max_contour, hull)

            with suppress(Exception):
                Lastend = end  # 1 - save current end position as copy.

            # image = cv2.putText(drawing, 'OpenCV', end, font, 1, (200, 0, 0), 3, cv2.LINE_AA)
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i
                res = contours[ci]
                hull = cv2.convexHull(res)
                if maxArea >= MinCountourSize:
                    cv2.drawContours(frame, [res], 0, (0, 255, 0), 2)
                    cv2.drawContours(frame, [hull], 0, (0, 0, 255), 3)

                with suppress(Exception):
                    isFinishCal, cnt = calculateFingers(res, drawing)

                # calculate moments of binary image
                M = cv2.moments(mask)

                if draw is True:
                    with suppress(Exception):
                        lastend = center  # 1 - save current end position as copy.
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                center = cX, cY
                with suppress(Exception):
                    s = defects[:, 0][:, 0]
                # Highlight the center
                if maxArea >= MinCountourSize:
                    cv2.circle(frame, center, 5, (255, 0, 255), -1)

                # Draw
                if cnt >= 4:

                    cv2.circle(drawing, center, PenSize, PenColor, -1)  # 3 - make circle at that position.
                    if draw is True:
                        with suppress(Exception):
                            cv2.line(drawing, lastend, center, PenColor, PenSize * 2)

                    if draw is False:
                        print("Draw gesture found with " + str(cnt) + " fingers")
                        lastend = ()
                        draw = True
                        eraser = False

                # Eraser
                if cnt == 0:
                    cv2.circle(drawing, center, EraserSize, EraserColor, -1)
                    if eraser is False:
                        print("Erase gesture found with " + str(cnt) + " fingers")
                        lastend = ()
                        eraser = True
                        draw = False

                # Clear
                if not WhiteBK:
                    drawing = np.full(frame.shape, 255, dtype=np.uint8)
                    WhiteBK = True

                    # Show all images

        # cv2.imshow('Mask', mask)
        frameHorizontal = cv2.flip(frame, 1)
        # cv2.imshow('Frame', frameHorizontal)
        if not WhiteBK:
            drawing = np.full(frame.shape, 255, dtype=np.uint8)

            WhiteBK = True
        drawingHorizontal = cv2.flip(drawing, 1)
        # cv2.imshow('Drawing', drawingHorizontal)

        # create 3 separate BGRA images as our "layers"
        MarkerFrame = np.full(frame.shape, 255, dtype=np.uint8)


        with suppress(Exception):
            cv2.circle(MarkerFrame, center, 5, (255, 0, 255), -1)

        MarkerFrameHorizontal = cv2.flip(MarkerFrame, 1)
        #cv2.imshow("out.png", MarkerFrameHorizontal)

        Gesturnary2 = cv2.addWeighted(MarkerFrameHorizontal, 0.5, drawingHorizontal, 0.5, 0)

        Gesturnary = np.concatenate((frameHorizontal, Gesturnary2), axis=1)
        cv2.imshow('Gesturnary', Gesturnary)



    else:
        # replay mp4
        draw = False
        lastend = ()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
