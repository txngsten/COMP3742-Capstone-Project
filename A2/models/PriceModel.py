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

class PriceModel:
    def __init__(self, model_params: npt.NDArray = np.empty(0)):
        self.model_params: npt.NDArray = model_params
        self.model = ...

    def train(self, training_set: pd.DataFrame):
        ...

    def inference(self, instance: pd.DataFrame):
        ...
