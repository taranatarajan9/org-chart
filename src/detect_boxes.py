import cv2
import numpy as np

def detect_boxes(image_path):
    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur and detect edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4:  # Rectangles have 4 corners
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h

            if w > 30 and h > 15 and aspect_ratio < 5:  # filter small lines or weird shapes
                boxes.append((x, y, w, h))
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show image with boxes
    cv2.imshow("Detected Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return boxes
