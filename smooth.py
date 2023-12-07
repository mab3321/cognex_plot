import cv2
import numpy as np

def smooth_border(image_path, kernel_size=15):
    # Read the grayscale image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original_image is None:
        print("Error: Could not read the image.")
        return

    # Apply GaussianBlur to smooth the borders
    smoothed_image = cv2.GaussianBlur(original_image, (kernel_size, kernel_size), 0)

    # Display the original and smoothed images
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Smoothed Image", smoothed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the smoothed image if needed
    cv2.imwrite("smoothed_image.jpg", smoothed_image)

if __name__ == "__main__":
    # Provide the path to your grayscale image
    image_path = r"C:\Users\MAB\Downloads\MegaScanner\binary_image.jpg"

    # Adjust the kernel size for the blurring operation if needed
    kernel_size = 15

    smooth_border(image_path, kernel_size)
