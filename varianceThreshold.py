import numpy as np

from Dataset import Dataset

class VarianceThreshold:

    def __init__(self, threshold: float):
        
        if threshold < 0:
            raise ValueError("Threshold has to be 0 or higher")

        self.threshold = threshold

        self.variance = None


    def fit(self, dataset : Dataset) -> 'VarianceThreshold':
        
        self.variance = np.var(dataset.X, axis=0)

        return self

    def transform(self, dataset : Dataset) -> Dataset:
        
        X = dataset.X
        
        new_features = self.variance > self.threshold
        X = X[:,new_features]
        features = np.array(dataset.feature_names)[new_features]

        return Dataset(X=X,y=dataset.y, feature_names=list(features),label_name = dataset.label_name)

    def fit_transform(self, dataset : Dataset) -> Dataset:

        self.fit(dataset)
        res = self.transform(dataset)

        return res


