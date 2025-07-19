# analyzer.py
import numpy as np
from extract_tread_from_cognex_disparity import correct_depth_profile, crop_profile, extract_outer_edge
import csv
import cv2
import matplotlib.pyplot as plt

class TreadDepthAnalyzer:
    def __init__(self, calibration_model):
        self.calibration_model = calibration_model

    def extract_outer_edge(self, image):
        blurred = cv2.medianBlur(image, 5)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        top_edge = np.argmax(thresh, axis=0)
        for i in range(len(top_edge)):
            if thresh[top_edge[i], i] == 0:
                top_edge[i] = top_edge[i - 1] if i > 0 else 0
        return top_edge

    def crop_profile(self, profile):
        valid_indices = np.where(profile > 0)[0]
        if len(valid_indices) == 0:
            return profile, 0
        first_valid_index = valid_indices[0]
        return profile[first_valid_index:], first_valid_index

    def correct_depth_profile(self, depth_profile_mm):
        max_depth = np.max(depth_profile_mm)
        return max_depth - depth_profile_mm

    def plot_profile(self, depth_profile_mm, x_start_index=0, save_path=None):
        x = np.arange(x_start_index, x_start_index + len(depth_profile_mm))
        y = depth_profile_mm

        plt.figure(figsize=(16, 8))
        plt.plot(x, y, color='blue', linewidth=1)
        plt.xlabel("Column Index (Pixels)")
        plt.ylabel("Tread Depth (mm)")
        plt.title("Corrected Tread Depth Profile (mm)")
        plt.grid(True)

        y_min = np.floor(np.min(y) / 0.5) * 0.5
        y_max = np.ceil(np.max(y) / 0.5) * 0.5
        plt.yticks(np.arange(y_min, y_max + 0.5, 0.5))

        if save_path:
            plt.savefig(save_path)
            print(f"📈 Plot saved to {save_path}")
        plt.close()

    def save_profile(self, depth_profile_mm, x_start_index=0, filename="tread_profile.csv"):
        x = np.arange(x_start_index, x_start_index + len(depth_profile_mm))
        data = zip(x, depth_profile_mm)
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Column Index", "Tread Depth (mm)"])
            writer.writerows(data)
        print(f"💾 Tread profile saved to {filename}")

    def analyze_image(self, image_path, plot_output_path=None, csv_output_path=None):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"❌ Could not read image at {image_path}")
            return

        outer_profile = self.extract_outer_edge(image)
        cropped_profile, start_x = self.crop_profile(outer_profile)
        depth_profile_mm = self.calibration_model.calculate_depth(cropped_profile)
        corrected_depth_profile = self.correct_depth_profile(depth_profile_mm)

        if plot_output_path:
            self.plot_profile(corrected_depth_profile, x_start_index=start_x, save_path=plot_output_path)
        if csv_output_path:
            self.save_profile(corrected_depth_profile, x_start_index=start_x, filename=csv_output_path)

        return corrected_depth_profile

