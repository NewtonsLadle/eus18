import cv2
import numpy as np

class KidneyThresholder(object):
    """A class to encapsulate code that applies threshold and containment
    to kidney annotations"""


    def __init__(self, threshold):
        """Constructs object

        Takes known threshold in Hounsfield Units
        """

        self.threshold = threshold


    def apply_threshold(self, annotation, slc, include_outline):
        """Applies criteria to narrow positive pixels

        Takes
            a binary annotation as a numpy array (255 or 0)
            a single slice as a numpy array of size some factor of the size
            of the annotation
        Returns
            a binary image with containment and threshold applied
        """
        # Apply floodfill to outside
        annotation = annotation.astype(np.uint8)
        annotation[annotation!=0] = 255

        holes = annotation.copy()
        cv2.floodFill(holes, None, (0,0), 255)

        holes = cv2.bitwise_not(holes)
        if (include_outline):
            holes = cv2.bitwise_or(annotation, holes)

        ret = np.logical_and(
            holes != 0,
            cv2.blur(slc, (9,9)) > self.threshold
        ).astype(np.int8)

        ret = cv2.blur(ret.astype(np.float32), (3,3))

        return ret
