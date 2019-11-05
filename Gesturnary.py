import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Defining the range of color in HSV
    lower_hand = np.array([30, 150, 80])
    upper_hand = np.array([125, 255, 255])

    # Threshold the HSV image to only get hand colors
    mask = cv2.inRange(hsv, lower_hand, upper_hand)

    # Display
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
