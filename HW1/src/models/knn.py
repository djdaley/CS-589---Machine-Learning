import pandas as pd
import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, features, classes):
        features = features.reset_index(drop=True)
        # self.feature_min = features.min().to_numpy()
        # self.feature_max = features.max().to_numpy()
        # self.feature_range = self.feature_max - self.feature_min
        self.X = features.to_numpy(dtype=float)
        self.Y = classes.reset_index(drop=True)
        self.classes = set(classes)

    def __predict_one__(self, x):
        x = np.asarray(x, dtype=float)
        # Normalize x
        # x = (x - self.feature_min) / self.feature_range

        distances = np.linalg.norm(self.X - x, axis=1)
        distances_df = pd.DataFrame({'Distance' : distances}).sort_values('Distance')
        counts = {x : 0 for x in self.classes}
        for i in distances_df.index[:self.k]:
            class_i = self.Y[i]
            counts[class_i] += 1
            if counts[class_i] > self.k/2:
                return class_i
        # tie or 'winning' class isn't majority (>2 classes)
        max_votes = 0
        for c, v in enumerate(counts):
            if v > max_votes:
                return_class = c

        return return_class

    def predict(self, X):
        return X.apply(self.__predict_one__, axis=1)