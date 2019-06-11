import cv2 as cv
from config.local import *
import numpy as np
import pickle


class HistVision:
    def __init__(self):
        self._database = self.load_database()

    @staticmethod
    def calc_hist(image, mask=None):
        hist = cv.calcHist([cv.cvtColor(image, cv.COLOR_BGR2HSV)],
                           config["hist_channels"],
                           mask,
                           config["hist_size"],
                           config["hist_range"])
        return cv.normalize(hist, hist)

    @staticmethod
    def dump_database(obj):
        with open(config["hist_database_path"], "wb") as file:
            pickle.dump(obj, file)

    @staticmethod
    def load_database():
        with open(config["hist_database_path"], "rb") as file:
            return pickle.load(file)

    def generate_database(self):
        pass

    def search_best(self, query_hist):
        result = ["Init value", float('inf')]
        # loop over indexes DB and find lesser distance between query and records
        # lesser dist means more 'similar' images
        for key in self._database:
            d = self.chi2_distance(self._database[key], query_hist)
            if d < result[1]:
                result[0], result[1] = key, d

        return result
