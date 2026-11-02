"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: DemandModel.py
Date: 02-11-2026
Description: Class file for training and performing inference for next-hour demand predictions.
"""

import numpy as np
import numpy.typing as npt

import pandas as pd


class DemandModel:
    def __init__(self, model_params: npt.NDArray = np.empty(0)):
        self.model_params: npt.NDArray = model_params
        self.model = ...

    def train(self, training_set: pd.DataFrame):
        ...

    def inference(self, instance: pd.DataFrame):
        ...
