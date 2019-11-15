import numpy as np
import cv2
from contextlib import suppress
# Toggle between using camera or test video
Realcam = False

if Realcam == True:
    cap = cv2.VideoCapture(0)

if Realcam == False:
    cap = cv2.VideoCapture('Vid1.mp4')

WhiteBK = False

colorsArray = []

pMarkerPos = []

# PenMarker settings
pMarker = True    # Default setting
pMarkerSize = 20
pMarkerPos = []
pMarkerColor = (255,0,0)
pMarkerThick = -1

# EraserMarker settings
eMarker = True    # Default setting
eMarkerSize = 20
eMarkerColor = (0,0,0)
eMarkerThick = 1

# Pen settings
draw = True    # Default setting
PenSize = 4
PenColor = (0,0,0)

# Eraser settings
eraser = False    # Default setting
EraserSize = 20
EraserColor = (255,255,255)

Lastend = ()

key = cv2.waitKey(1)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # clear drawing array with C
    if (key & 0xFF == ord('c')):
        points = []
        WhiteBK = False

    # D to start drawing
    elif key & 0xFF == ord('s') and draw == False:
        draw = True
        eraser = False
        print("started drawing")

    # E to stop drawing
    if (key & 0xFF == ord('d')) and eraser == False:
        eraser = True
        draw = False
        print("started eraser")

    # E to start drawing
    elif key & 0xFF == ord('a'):
        eraser = False
        draw = False
        print("stopped all")

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

        # draw contour
        cv2.drawContours(frame, contours, -2, (0, 255, 0), 3)


        # bounding boxes and circles around countour
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red rectangle
            #cv2.drawContours(frame, [box], 0, (0, 0, 255))

            # finally, get the min enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(c)
            # convert all values to int
            center = (int(x), int(y))
            radius = int(radius)
            # and draw the circle in blue
            img = cv2.circle(frame, center, radius, (255, 0, 0), 2)
            # highlight center
            cv2.circle(frame, ((int(x),int(y))), 5, (255, 0, 0), -1)

        if contours:
            # assume largest contour is the one of interest
            max_contour = max(contours, key=cv2.contourArea)

            epsilon = 0.01 * cv2.arcLength(max_contour, True)
            max_contour = cv2.approxPolyDP(max_contour, epsilon, True)

            # determine convex hull & convexity defects of the hull
            hull = cv2.convexHull(max_contour, returnPoints=False)
            defects = cv2.convexityDefects(max_contour, hull)

            with suppress(Exception):
                Lastend = end     # 1 - save current end position as copy.

            # image = cv2.putText(drawing, 'OpenCV', end, font, 1, (200, 0, 0), 3, cv2.LINE_AA)


            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])   # 2 - get new end position.
                    far = tuple(max_contour[f][0])
                    cv2.line(frame, start, end, [255, 255, 0], 2)
                    cv2.circle(frame, end, 5, [0, 0, 255], -1)

                # draw line from center to end
                    cv2.line(frame, center, end, [255, 255, 0], 2)

                    c = 0
                    X = 0
                    Y = 0
                    # white background on "drawing"

                    if WhiteBK is False:
                        drawing = np.full(frame.shape, 255, dtype=np.uint8)
                        WhiteBK = True
                    key = cv2.waitKey(1)

                    # Draw- draw circle at drawing point in "drawing"
                    # Pen
                    if draw is True:
                        cv2.circle(drawing, end, PenSize, PenColor, -1)  # 3 - make circle at that position.
                        with suppress(Exception):
                            cv2.line(drawing, Lastend, end, PenColor, PenSize*2)


                    # Eraser
                    if eraser is True:
                        cv2.circle(drawing, end, EraserSize, EraserColor, -1)


                    # Show all images
        cv2.imshow('Mask', mask)
        cv2.imshow('Frame', frame)
        cv2.imshow('Drawing', drawing)

    else:
        # replay mp4
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()