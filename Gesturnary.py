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
    lower_hand = np.array([30, 150, 80])
    upper_hand = np.array([125, 255, 255])

    # Threshold the HSV image to only get hand colors
    mask = cv2.inRange(hsv, lower_hand, upper_hand)

    # Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # Display
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# test test
