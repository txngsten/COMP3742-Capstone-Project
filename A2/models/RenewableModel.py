"""
Student Name:
Student FAN:
File:
Date:
Description:
"""

import numpy as np
import numpy.typing as npt

import pandas as pd

class RenewableModel:
    def __init__(self, model_params: npt.NDArray = np.empty(0)):
        self.model_params: npt.NDArray = model_params
        self.can_predict: bool = model_params.size != 0

    def train(self, training_set: pd.DataFrame):
        ...

    def inference(self, instance: pd.DataFrame):
        ...
