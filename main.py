import cv2
import numpy as np
def create_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])  
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

def centroid( contour):
    if len(contour) == 0:
        return (-1, -1)
    M = cv2.moments(contour)
    try:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
    except ZeroDivisionError:
        return (-1, -1)
    return (x, y)

def find_contours( thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
cap=cv2.VideoCapture(0)
def get_centroid(frame):
    mask = create_mask(frame)
    contours = find_contours(mask)
    centroids = []
    for contour in contours:
        cx, cy =centroid(contour)
        centroids.append((cx, cy))
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    return centroids
while True:
    sucess,frame=cap.read()
    if not sucess:
        break
    frame=cv2.flip(frame,1)
    centroids = get_centroid(frame)
    cv2.imshow("Leaf Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()