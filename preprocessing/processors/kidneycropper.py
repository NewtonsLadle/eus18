import cv2
import numpy as np
import sys
import math

#Attempt 2: Runs much faster and more accurately, flood fill is now functional

#Takes an image returns a double representing the roundness of a contour
#1.0 = pefect circle
#returns Area / ((Perimeter^2) / 4PI)
def roundness(annotation):
    gray = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 20, 255, 0)
    contour_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = contours[0]
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    adjusted_perimeter = (perimeter ** 2) / (4 * math.pi)
    return area /adjusted_perimeter

#Returns distance between two Points
def point_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1]) ** 2)**0.5

# Fills in center with flood fill
def binary_flood_fill(binary_image):
    flood = binary_image.copy()
    height, width = flood.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255);
    flood_inverse = cv2.bitwise_not(flood)

    fully_flooded = binary_image | flood_inverse

    return fully_flooded

#Takes an image and draws a line across the two major points surrounding the hilum, and flood fills the center
def crop_kidney(annotation):
    result = annotation

    gray = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 20, 255, 0)
    contour_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    primary_contour = contours[0]

    #Finds the convexities and draws a line across the longest one (the most concave point)
    hull = cv2.convexHull(primary_contour, returnPoints=False)
    defects = cv2.convexityDefects(primary_contour, hull)
    longest_distance = 0
    start_point, end_point = (0, 0)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(primary_contour[s][0])
        end = tuple(primary_contour[e][0])
        if point_distance(start, end) > longest_distance:
            longest_distance = point_distance(start, end)
            start_point = start
            end_point = end

    cv2.line(contour_image, start_point, end_point, [255, 255, 255], 2)

    flooded = binary_flood_fill(thresh)


    return flooded



if __name__ == '__main__':
    # Get image location from command line
    if(len(sys.argv) > 1):
        image = cv2.imread(sys.argv[1])
        #only compute images with less than 0.55 roundness
        if(roundness(image) < 0.55):
            output = crop_kidney(image)
            # Save to disk
            cv2.imwrite("completed-"+sys.argv[1], output)
