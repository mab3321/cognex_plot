import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create a binary image for illustration purposes
binary_image = np.random.randint(0, 256, size=(300, 400), dtype=np.uint8)
binary_image[binary_image > 60] = 255
binary_image[binary_image <= 60] = 0

# Find coordinates where pixel values are greater than 200
coordinates = np.where(binary_image > 200)

# Display the binary image
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.show()

# Print the coordinates
print("Coordinates where pixel values are greater than 200:")
print(coordinates)