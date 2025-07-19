import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_outer_edge(image):
    """
    Extract the outer (topmost) edge profile of the bright region in the image.
    """
    blurred = cv2.medianBlur(image, 5)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    top_edge = np.argmax(thresh, axis=0)

    for i in range(len(top_edge)):
        if thresh[top_edge[i], i] == 0:
            top_edge[i] = top_edge[i - 1] if i > 0 else 0

    return top_edge

def crop_profile(profile):
    """
    Crop the profile to remove leading zero values (invalid/no data).
    """
    valid_indices = np.where(profile > 0)[0]
    if len(valid_indices) == 0:
        return profile, 0

    first_valid_index = valid_indices[0]
    return profile[first_valid_index:], first_valid_index

def get_constant_for_position(x, total_length, constants=(0.21, 0.22, 0.23)):
    """
    Return region-based constant depending on the X position.
    """
    region_length = total_length // 3
    if x < region_length:
        return constants[0]
    elif x < 2 * region_length:
        return constants[1]
    else:
        return constants[2]

def calculate_tread_depth_mm_region(profile, constants=(0.21, 0.22, 0.23)):
    """
    Convert profile to tread depth in mm relative to outer edge using region based constants.
    """
    outer_edge_value = np.min(profile)
    pixel_gaps = profile - outer_edge_value

    total_length = len(profile)
    depth_mm = np.zeros_like(profile, dtype=np.float32)

    for idx, pixel_gap in enumerate(pixel_gaps):
        constant = get_constant_for_position(idx, total_length, constants)
        depth_mm[idx] = pixel_gap * constant

    return depth_mm

def correct_depth_profile(depth_profile_mm):
    """
    Correct the depth profile so that outer ring is 0 mm and grooves go upwards.
    """
    max_depth = np.max(depth_profile_mm)
    corrected_depth = max_depth - depth_profile_mm
    return corrected_depth

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

    if save_path:
        plt.savefig(save_path)
        print(f"📈 Plot saved to {save_path}")
    plt.close()

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
    depth_profile_mm = calculate_tread_depth_mm_region(cropped_profile, constants=(0.21, 0.22, 0.23))
    corrected_depth_profile = correct_depth_profile(depth_profile_mm)
    plot_profile_mm(corrected_depth_profile, title="Corrected Tread Depth Profile (mm)", x_start_index=start_x, save_path=plot_output_path)

# analyze_and_plot_tread_profile(img_path,"new.png")
