from typing import Callable
import numpy as np
from f_classif import f_classif
from f_regression import f_regression
from Dataset import Dataset

class SelectKBest:
    def __init__(self, score_func: Callable = f_classif, k: int = 10):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset):
        
        self.F, self.p = self.score_func(dataset)
        

    def transform(self, dataset:Dataset) -> Dataset:
        ids = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.feature_names)[ids]

        d = Dataset()
        d.setDataset(x=dataset.X[:,ids], y=dataset.y, feature_names=list(features), label_name=dataset.label_name)

        return d

    def fit_transform(self , dataset: Dataset) -> Dataset:
        self.fit(dataset)
        res = self.transform(dataset)

        return res


if __name__ == '__main__':
    
    dataset = Dataset()
    dataset.load(filename='numeros.csv')

    # Create an instance of SelectKBest
    select_k_best = SelectKBest(k=2)

    # Fit and transform the dataset
    transformed_dataset = select_k_best.fit_transform(dataset)

    # Print the transformed dataset
    print(transformed_dataset.X)
    print(transformed_dataset.y)
    print(transformed_dataset.feature_names)
    print(transformed_dataset.label_name)