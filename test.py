import serial

ser = serial.Serial('COM4', 9600, timeout=1)
threshold = 3.0
was_below_threshold = False  # Keeps track of previous state

def read_voltage():
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                try:
                    voltage = float(line)
                    if voltage < threshold:
                        if not was_below_threshold:
                            print("⚠️  Detected: Voltage below threshold!")
                            was_below_threshold = True
                    else:
                        was_below_threshold = False  # Reset when voltage rises
                except ValueError:
                    print("Invalid voltage format")
        except KeyboardInterrupt:
            print("\nStopped by user")
            break
import os
import sys

# Add HALCON DLL path
halcon_bin_path = r"C:\Users\DELL\AppData\Local\Programs\MVTec\HALCON-24.11-Progress-Student-Edition\bin\x64-win64"
os.environ["PATH"] = halcon_bin_path + os.pathsep + os.environ["PATH"]
import halcon as ha
import cv2
import numpy as np

def main():
    acq_handle = ha.open_framegrabber(
        'GigEVision2',
        1, 1, 0, 0, 0, 0,
        'progressive', -1, 'default', -1, 'false', 'default',
        '00d0243dd49c_CognexCorporation_DS1300', 0, -1
    )

    image_counter = 0
    print("📸 Press SPACE to capture. ESC to exit.")

    # Show dummy window to capture keyboard input
    cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
    cv2.imshow("Input", 255 * np.ones((100, 400), dtype=np.uint8))

    try:
        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC
                print("🛑 Exit requested.")
                break
            elif key == 32:  # SPACE
                print("📷 Grabbing image...")

                try:
                    image, region, contours, data = ha.grab_data_async(acq_handle, 5000)
                    filename = f"capture_{image_counter:03d}.png"
                    ha.write_image(image, 'png', 0, filename)
                    print(f"✅ Saved: {filename}")
                    image_counter += 1
                except ha.HOperatorError:
                    print("⚠️ Sensor did not respond or timed out.")

    finally:
        ha.close_framegrabber(acq_handle)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    read_voltage()
