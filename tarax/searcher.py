import numpy as np
import csv


class Searcher:
    def __init__(self, indexes_path):
        self._indexes_path = indexes_path
        self._indexes = {}
        self._load_indexes()

    def _load_indexes(self):
        with open(self._indexes_path) as file:
            reader = csv.reader(file)
            for row in reader:
                self._indexes[row[0]] = [float(x) for x in row[1:]]

    def search_best(self, query_features):
        result = ["Init value", float('inf')]
        # loop over indexes DB and find lesser distance between query and records
        # lesser dist means more 'similar' images
        for key in self._indexes:
            d = self.chi2_distance(self._indexes[key], query_features)
            if d < result[1]:
                result[0], result[1] = key, d

        return result

    def chi2_distance(self, hist_a, hist_b, eps=1e-10):
        # compute the chi-squared distance
        return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(hist_a, hist_b)])
