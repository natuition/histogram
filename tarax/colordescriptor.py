import numpy as np
import cv2


class ColorDescriptor:
    def __init__(self, bins):
        # store the number of bins for the 3D histogram
        self._bins = bins

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        c_x, c_y = int(w * 0.5), int(h * 0.5)  # center x, y
        features = []
        corner_mask = np.zeros(image.shape[:2], dtype="uint8")

        # loop over the segments
        # list contains coordinates for 4 image rectangle segments corners
        for (start_x, start_y, end_x, end_y) in [(0, 0, c_x, c_y), (c_x, 0, w, c_y), (0, c_y, c_x, h), (c_x, c_y, w, h)]:
            # construct a mask for each corner of the image, subtracting
            corner_mask.fill(0)
            cv2.rectangle(corner_mask, (start_x, start_y), (end_x, end_y), 255, -1)
            features.extend(self.histogram(image, corner_mask))

        return features

    def histogram(self, image, mask):
        """Extract a 3D color histogram from the masked region of the
        image, using the supplied number of bins per channel; then
        normalize the histogram"""

        hist = cv2.calcHist([image], [0, 1, 2], mask, self._bins, [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()
