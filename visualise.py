import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_image(file_path):
    try:
        # Load the image
        img = mpimg.imread(file_path)

        # Display the image
        plt.imshow(img)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Display an image from the command line.")
    parser.add_argument("--image_path", default="image.jpg", help="Path to the image file")

    args = parser.parse_args()
    display_image(args.image_path)

if __name__ == "__main__":
    main()

