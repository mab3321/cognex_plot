#/models/calibration_model.py
import numpy as np

class CalibrationModel:
    def __init__(self, params):
        self.params = {
            "left": (params["m_left"], params["b_left"]),
            "middle": (params["m_middle"], params["b_middle"]),
            "right": (params["m_right"], params["b_right"])
        }