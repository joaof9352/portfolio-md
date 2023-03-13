import numpy as np
import scipy as stats
from typing import Tuple, Union
from Dataset import Dataset

def f_classif(dataset: Dataset) -> Union[Tuple[np.ndarray,np.ndarray],Tuple[float,float]]:

    classes = dataset.get_classes() #temos de sacar as classes do dataset
    grupos = [dataset.X[dataset.y == c] for c in classes] #os grupos por classe que presumo que seja a unica existente noo dataset.py

    F, p = stats.f_oneway(grupos)

    return F, p
