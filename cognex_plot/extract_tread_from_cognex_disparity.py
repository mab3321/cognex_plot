import cv2
import numpy as np
import matplotlib.pyplot as plt
from constants import thresholds

# Continuous calibration model based on your data
import numpy as np

# Coefficients from least squares fit (based on your calibration data)
# These are from the previous fitting step; you can refine with more data if needed

m_left, b_left = -0.000427, 0.4908
m_middle, b_middle = -0.000609, 0.6443
m_right, b_right = -0.000618, 0.6557

def get_continuous_constants(y):
    """
    Get per-region constants based on Y (row) using continuous linear models.
    """
    c_left = m_left * y + b_left
    c_middle = m_middle * y + b_middle
    c_right = m_right * y + b_right
    return (c_left, c_middle, c_right)


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

def get_constant_for_position(x, y_value, total_length, y_threshold=730,
                               constants_below=(0.21, 0.22, 0.23),
                               constants_above=(0.11, 0.12, 0.13)):
    """
    Return region-based constant depending on X and Y position.
    If the detected Y position is after y_threshold, use different constants.
    """
    if y_value >= y_threshold:
        constants = constants_above
        if x == 0:
            print(f"⚙️ Using constants ABOVE threshold {y_threshold}: {constants_above}")
    else:
        constants = constants_below
        if x == 0:
            print(f"⚙️ Using constants BELOW threshold {y_threshold}: {constants_below}")

    region_length = len(constants)  # assume 3 regions
    part_length = total_length // region_length

    for i in range(region_length):
        if x < (i + 1) * part_length:
            return constants[i]

    return constants[-1]


def get_profile_region_constant(profile_middle_value, thresholds):
    """
    Select constants based on the Y-position of the profile's middle point.
    """
    for threshold, constants in thresholds:
        if profile_middle_value >= threshold:
            print(f"📏 Profile middle Y={profile_middle_value} -> Using constants {constants} (threshold={threshold})")
            return constants
    # Fallback if none match (shouldn't happen if 0 is in thresholds)
    return thresholds[-1][1]


def calculate_tread_depth_mm_region(profile):
    """
    Convert profile to tread depth in mm using continuous Y-dependent constants.
    Prints the middle Y value and the constants used.
    """
    outer_edge_value = np.min(profile)
    pixel_gaps = profile - outer_edge_value

    total_length = len(profile)
    depth_mm = np.zeros_like(profile, dtype=np.float32)

    # Use middle of the profile to determine the Y-dependent constants
    middle_idx = total_length // 2
    profile_middle_value = profile[middle_idx]

    # Debug print: Show middle Y coordinate
    print(f"📍 Middle index: {middle_idx}, Y value at middle: {profile_middle_value}")

    c_left, c_middle, c_right = get_continuous_constants(profile_middle_value)

    # Debug print: Show the constants being used
    print(f"⚙️ Using continuous constants at Y={profile_middle_value}: Left={c_left:.4f}, Middle={c_middle:.4f}, Right={c_right:.4f}")

    # Apply per-region constants (image-based: left, middle, right)
    for idx, pixel_gap in enumerate(pixel_gaps):
        if idx < total_length // 3:
            constant = c_left
        elif idx < 2 * total_length // 3:
            constant = c_middle
        else:
            constant = c_right
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
    depth_profile_mm = calculate_tread_depth_mm_region(cropped_profile)
    corrected_depth_profile = correct_depth_profile(depth_profile_mm)
    plot_profile_mm(corrected_depth_profile, title="Corrected Tread Depth Profile (mm)", x_start_index=start_x, save_path=plot_output_path)

# analyze_and_plot_tread_profile(img_path,"new.png")
