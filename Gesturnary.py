import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Blur image
    blurred_frame = cv2.bilateralFilter(frame,9,75,75)



    # Convert from BGR to HSV
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Defining the range of color in HSV
    # Green
    lower_hand = np.array([40, 100, 80])
    upper_hand = np.array([80, 255, 255])

     #lower_hand = np.array([30, 150, 80])
     #upper_hand = np.array([125, 255, 255])

    # Blue
    # lower_hand = np.array([38, 85, 0])
    # upper_hand = np.array([120, 255, 255])

    # Threshold the HSV image to only get hand colors
    mask = cv2.inRange(hsv, lower_hand, upper_hand)

    # Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    # Display
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
