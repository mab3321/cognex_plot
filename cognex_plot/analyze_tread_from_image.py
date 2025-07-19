import cv2
import numpy as np
import matplotlib.pyplot as plt

from extract_tread_from_cognex_disparity import calculate_tread_depth_mm_region, correct_depth_profile, crop_profile, extract_outer_edge


def plot_profile_mm(depth_profile_mm, title="Corrected Tread Depth Profile (mm)", x_start_index=0, save_path=None):
    """
    Plot corrected tread depth profile in mm and save to disk.
    """
    x = np.arange(x_start_index, x_start_index + len(depth_profile_mm))
    y = depth_profile_mm

    plt.figure(figsize=(16, 8))
    plt.plot(x, y, color='blue', linewidth=1)
    plt.xlabel("Column Index (Pixels)")
    plt.ylabel("Tread Depth (mm)")
    plt.title(title)
    plt.grid(True)

    y_min = np.floor(np.min(y) / 0.5) * 0.5
    y_max = np.ceil(np.max(y) / 0.5) * 0.5
    plt.yticks(np.arange(y_min, y_max + 0.5, 0.5))

    plt.show()

# ==== Example usage ====

# Load image
img_path = "captures\\laser_trigger_000.png"
def analyze_and_plot_tread_profile(image_path, plot_output_path=None):
    """
    Analyze tread profile and save plot to file.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Could not read image at {image_path}")
        return

    outer_profile = extract_outer_edge(image)
    cropped_profile, start_x = crop_profile(outer_profile)
    depth_profile_mm = calculate_tread_depth_mm_region(cropped_profile)
    corrected_depth_profile = correct_depth_profile(depth_profile_mm)
    plot_profile_mm(corrected_depth_profile, title="Corrected Tread Depth Profile (mm)", x_start_index=start_x, save_path=plot_output_path)

analyze_and_plot_tread_profile(img_path,"new.png")
