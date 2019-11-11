import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Blur image
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert from BGR to HSV
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Defining the range of color in HSV
    # Green
    lower_hand = np.array([30, 150, 80])
    upper_hand = np.array([125, 255, 255])
    # Blue
    # lower_hand = np.array([38, 85, 0])
    # upper_hand = np.array([120, 255, 255])

    # Threshold the HSV image to only get hand colors
    mask = cv2.inRange(hsv, lower_hand, upper_hand)

    # Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -2, (0, 255, 0), 3)

    if contours:
        cnt = contours[0]
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # hull = cv2.convexHull(cnt)

        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                cv2.line(frame, start, end, [255, 255, 0], 2)
                cv2.circle(frame, far, 5, [0, 0, 255], -1)

    # Display
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
