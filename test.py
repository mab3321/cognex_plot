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

def calculate_tread_depth_mm(profile, constant_mm_per_pixel=0.22):
    """
    Convert profile to tread depth in mm relative to outer edge (which is zero).
    """
    outer_edge_value = np.min(profile)
    pixel_gaps = profile - outer_edge_value
    depth_mm = pixel_gaps * constant_mm_per_pixel
    return depth_mm

def correct_depth_profile(depth_profile_mm):
    """
    Correct the depth profile so that outer ring is 0 mm and grooves go upwards.
    """
    max_depth = np.max(depth_profile_mm)
    corrected_depth = max_depth - depth_profile_mm
    return corrected_depth

def plot_profile_mm(depth_profile_mm, title="Corrected Tread Depth Profile (mm)", x_start_index=0):
    """
    Plot corrected tread depth profile in mm.
    """
    x = np.arange(x_start_index, x_start_index + len(depth_profile_mm))
    y = depth_profile_mm

    plt.figure(figsize=(16, 8))
    plt.plot(x, y, color='blue', linewidth=1)
    plt.xlabel("Column Index (Pixels)")
    plt.ylabel("Tread Depth (mm)")
    plt.title(title)
    plt.grid(True)
    plt.show()

# ==== Example usage ====

# Load image
img_path = "saved_image_000.png"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Extract outer edge
outer_profile = extract_outer_edge(image)

# Step 2: Remove leading zeros
cropped_profile, start_x = crop_profile(outer_profile)

# Step 3: Calculate tread depth in mm
depth_profile_mm = calculate_tread_depth_mm(cropped_profile, constant_mm_per_pixel=0.22)

# Step 4: Correct depth profile (outer ring = 0 mm, grooves upwards)
corrected_depth_profile = correct_depth_profile(depth_profile_mm)

# Step 5: Plot corrected tread depth profile
plot_profile_mm(corrected_depth_profile, title="Corrected Tread Depth Profile in mm", x_start_index=start_x)
