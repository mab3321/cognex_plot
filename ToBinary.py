import cv2
import numpy as np

def convert_to_binary(image_path, threshold_value=128):
    # Read the grayscale image
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        print("Error: Could not read the image.")
        return

    # Apply binary thresholding
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Display the original and binary images
    cv2.imshow("Grayscale Image", gray_image)
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the binary image if needed
    cv2.imwrite("binary_image.jpg", binary_image)

if __name__ == "__main__":
    # Provide the path to your grayscale image
    image_path = r"C:\Users\MAB\Downloads\MegaScanner\smoothed_image.jpg"

    # Adjust the threshold value for binary conversion if needed
    threshold_value = 128

    convert_to_binary(image_path, threshold_value)
