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

def red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    return red_mask

def detect_red_in_leaves(contour, frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    leaf_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(leaf_mask, [contour], -1, (0,0,255), -1) 

    red_in_leaf = cv2.bitwise_and(red_mask, red_mask, mask=leaf_mask)

    red_pixels = cv2.countNonZero(red_in_leaf)
    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    if total_leaf_pixels == 0:
        return 0  
    red_percent = (red_pixels / total_leaf_pixels) * 100
    return red_percent

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

def find_contours(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

url="http://10.24.186.79:8080/video"
cap=cv2.VideoCapture(0)
def get_centroid(frame):
    mask = create_mask(frame)
    contours = find_contours(mask)
    centroids = []
    total=0
    for contour in contours:
        total+=detect_red_in_leaves(contour, frame)
        cx, cy =centroid(contour)
        centroids.append((cx, cy))
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    print(f"Total red percentage in leaves: {total:.2f}%")
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