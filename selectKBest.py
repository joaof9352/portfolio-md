from typing import Callable
import numpy as np
import f_classif
from Dataset import Dataset

class SelectKBest:
    def __init__(self, score_func: Callable = f_classif, k: int = 10):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        
        self.F, self.p = self.score_func(dataset)
        
        return self

    def transform(self, dataset:Dataset) -> Dataset:
        ids = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.feature_names)[ids]

        return Dataset(X=dataset.X[:,ids], y=dataset.y, feature_names=list(features), label_name=dataset.label_name)


    
    
    def fit_transform(self , dataset: Dataset) -> Dataset:
        self.fit(dataset)
        res = self.transform(dataset)

        return res