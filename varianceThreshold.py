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

        d = Dataset()
        d.setDataset(x=dataset.X, y=dataset.y, feature_names=list(features), label_name=dataset.label_name)

        return d

    def fit_transform(self, dataset : Dataset) -> Dataset:

        self.fit(dataset)
        res = self.transform(dataset)

        return res

if __name__ == '__main__':
    
    dataset = Dataset()
    dataset.load(filename='numeros.csv')

    # Create an instance of SelectKBest
    threshold = 1.0
    variance_threshold = VarianceThreshold(threshold=threshold)

    # Fit and transform the dataset
    transformed_dataset = variance_threshold.fit_transform(dataset)

    # Print the transformed dataset
    print(transformed_dataset.feature_names)
    print(transformed_dataset.X)


