import cv2
import numpy as np
import sys
import math

#Attempt 2: Runs much faster and more accurately, flood fill is now functional


class HilumFiller(object):
    """A processor which takes an annotation and fills the hilum in"""

    def __init__(self):
        """Construct filler object"""
        self.roundness_threshold = 0.90

    #Takes an image returns a double representing the roundness of a contour
    #1.0 = pefect circle
    #returns Area / ((Perimeter^2) / 4PI)
    def _roundness(self, annotation):
        # gray = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
        gray = np.uint8(annotation)
        ret, thresh = cv2.threshold(gray, 0, 255, 0)
        contour_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours) > 0):
            main_contour = contours[0]
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            adjusted_perimeter = (perimeter ** 2) / (4 * math.pi)
            if (adjusted_perimeter > 1e-5):
                return area /adjusted_perimeter
            else:
                return 0
        else:
            return 0

    #Returns distance between two Points
    def _point_distance(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1]) ** 2)**0.5

    # Fills in center with flood fill
    def _binary_flood_fill(self, binary_image):
        flood = binary_image.copy()
        height, width = flood.shape[:2]
        mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255);
        flood_inverse = cv2.bitwise_not(flood)

        fully_flooded = binary_image | flood_inverse

        return fully_flooded

    #Takes an image and draws a line across the two major points surrounding the hilum, and flood fills the center
    def fill(self, annotation):
        # gray = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
        gray = np.uint8(annotation)
        ret, thresh = cv2.threshold(gray, 0, 255, 0)

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
        sizes = stats[:, -1]

        max_label = 0
        max_size = 0
        for i in range(1, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        thresh[output != max_label] = 0

        contour_image, contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )


        if (len(contours) == 0):
            return annotation, False


        if(self._roundness(thresh) > self.roundness_threshold):
            flooded = self._binary_flood_fill(thresh)
            return (flooded > 0).astype(np.uint8), True


        primary_contour = contours[0]


        #Finds the convexities and draws a line across the longest one (the most concave point)
        hull = cv2.convexHull(primary_contour, returnPoints=False)
        defects = cv2.convexityDefects(primary_contour, hull)
        longest_distance = 0
        start_point, end_point = (0, 0)

        if (defects is not None):
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(primary_contour[s][0])
                end = tuple(primary_contour[e][0])
                if self._point_distance(start, end) > longest_distance:
                    longest_distance = self._point_distance(start, end)
                    start_point = start
                    end_point = end

            cv2.line(contour_image, start_point, end_point, [255, 255, 255], 2)

        flooded = self._binary_flood_fill(thresh)
        return (flooded > 0).astype(np.uint8), True
