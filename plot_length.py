import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Constants
constant = 0.0641
image_path = 'image.jpg'

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale and apply smoothing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # Apply median filtering for noise reduction

    # Apply adaptive thresholding for better boundary detection
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

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

def smooth_and_plot(upper_border_filled):
    # Smooth the upper border
    y_smoothed = np.convolve(upper_border_filled, np.ones(50) / 50, mode='valid') * constant

    # Find the peaks with prominence to filter out smaller peaks - adjusted parameters for this specific case
    peaks, properties = find_peaks(y_smoothed, prominence=0.3)  # Adjust prominence as needed

    # Plot the updated graph with annotations for the peaks
    plt.figure(figsize=(14, 7))
    plt.plot(y_smoothed, label='Smoothed Upper Border')
    plt.plot(peaks, y_smoothed[peaks], "x", label='Peaks')

    # Annotate the peaks with their y-values (transformed to mm)
    for peak, value in zip(peaks, y_smoothed[peaks]):
        plt.annotate(f'{value:.2f}', (peak, value), textcoords="offset points", 
                     xytext=(-15,10), ha='center', 
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.title('Peak and Trough Differences')
    plt.xlabel("X axis")
    plt.ylabel("Transformed Y axis (mm)")
    plt.legend()
    plt.show()

# Main program
thresh = preprocess_image(image_path)
upper_border = find_upper_border(thresh)
upper_border_filled = fill_gaps_with_previous(upper_border)
smooth_and_plot(upper_border_filled)
