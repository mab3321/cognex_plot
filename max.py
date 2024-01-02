import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Since the actual image data isn't available, we'll use a mock signal for illustration.
# Let's assume 'y' contains the values of the 'upper_border_smoothed' from your plot.
# Replace this with 'upper_border_smoothed' when running your actual data.
y = np.array([385, 390, 395, 400, 405, 395, 385, 390, 405, 410, 415, 420,
              410, 400, 390, 380, 390, 400, 410, 405, 395, 385])

# Find peaks (maxima) and troughs (minima)
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