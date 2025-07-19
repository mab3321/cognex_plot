# tread_depth_analyzer.py

import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from analyzer import TreadDepthAnalyzer

from config import M_LEFT, B_LEFT, M_MIDDLE, B_MIDDLE, M_RIGHT, B_RIGHT
from models.calibration_model import CalibrationModel
# Example usage
if __name__ == "__main__":
    calib_model = CalibrationModel(M_LEFT, B_LEFT, M_MIDDLE, B_MIDDLE, M_RIGHT, B_RIGHT)
    analyzer = TreadDepthAnalyzer(calib_model)

    img_path = "captures/laser_trigger_000.png"
    analyzer.analyze_image(img_path, plot_output_path="tread_plot.png", csv_output_path="tread_profile.csv")
