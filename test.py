import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale and apply smoothing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # Apply median filtering for noise reduction

    # Apply adaptive thresholding for better boundary detection
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    return thresh

def find_upper_border(thresh):
    # Find the uppermost non-zero pixels in each column using Canny edge detector
    edges = cv2.Canny(thresh, 50, 150)  # Adjust the threshold values

    # Apply dilation to close small gaps
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Apply closing to fill small gaps
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find upper border
    upper_border = np.argmax(edges, axis=0)

    return upper_border

def fill_gaps_with_previous(upper_border):
    # Fill gaps with the same number as the previous non-zero value
    for i in range(1, len(upper_border)):
        if upper_border[i] == 0:
            upper_border[i] = upper_border[i - 1]

    return upper_border

def smooth_and_plot(upper_border):
    # Smooth the upper border
    window_size = 25
    y = np.convolve(upper_border, np.ones(window_size) / window_size, mode='valid')

    peaks, _ = find_peaks(y)
    troughs, _ = find_peaks(-y)

    # Calculate differences between consecutive extrema (peaks and troughs)
    peak_diffs = np.diff(peaks)
    trough_diffs = np.diff(troughs)

    # For illustrative purposes, let's print the results.
    # In your actual code, you would use these values to annotate your plot.
    print("Peak differences:", peak_diffs)
    print("Trough differences:", trough_diffs)

    # Plot the mock signal and the peaks and troughs.
    plt.plot(y, label='Smoothed Upper Border')
    plt.plot(peaks, y[peaks], "x", label='Peaks')
    plt.plot(troughs, y[troughs], "o", label='Troughs')
    plt.title('Peak and Trough Differences')
    plt.legend()
    plt.show()

    # Output the differences as a dictionary (as an example).
    # You can modify this to output in a format suitable for your application.
    extrema_diffs = {
        'peak_diffs': peak_diffs.tolist(),
        'trough_diffs': trough_diffs.tolist()
    }

    print(extrema_diffs)
    # # Find extrema
    # extrema = np.diff(np.sign(np.diff(upper_border_smoothed)))
    # minima_indices = np.where(extrema == -2)[0] + 1  # Add 1 to account for diff
    # maxima_indices = np.where(extrema == 2)[0] + 1

    # # Calculate differences between consecutive relative extrema
    # minima_differences = np.diff(linear_interpolation(minima_indices))
    # maxima_differences = np.diff(linear_interpolation(maxima_indices))

    # # Convert differences to millimeters (multiply by 0.108)
    # minima_differences_mm = minima_differences * 0.108
    # maxima_differences_mm = maxima_differences * 0.108

    # # Plot the linear interpolation with transformed y-axis
    # plt.plot(x_interp, linear_interpolation(x_interp) * 0.108, color='black')

    # for idx, diff in zip(minima_indices[:-2], minima_differences_mm):
    #     plt.annotate(f'{diff:.2f} mm', (idx, linear_interpolation(idx) * 0.108), textcoords="offset points", xytext=(-10,-10), ha='center')

    # for idx, diff in zip(maxima_indices[:-2], maxima_differences_mm):
    #     plt.annotate(f'{diff:.2f} mm', (idx, linear_interpolation(idx) * 0.108), textcoords="offset points", xytext=(-10,10), ha='center')

    # plt.xlabel("X axis")
    # plt.ylabel("Tread Depth (mm)")  # Update y-axis label
    # plt.title("Smoothed Upper Border with Gaps Filled")
    # plt.show()

# Main program
image_path = "image.jpg"  # Replace with your image path
thresh = preprocess_image(image_path)
upper_border = find_upper_border(thresh)
upper_border_filled = fill_gaps_with_previous(upper_border)
smooth_and_plot(upper_border_filled)
