import os
import time
from extract_tread_from_cognex_disparity import analyze_and_plot_tread_profile

# Set HALCON environment path
import sys
halcon_bin_path = r"C:\Users\DELL\AppData\Local\Programs\MVTec\HALCON-24.11-Progress-Student-Edition\bin\x64-win64"
os.environ["PATH"] = halcon_bin_path + os.pathsep + os.environ["PATH"]
import halcon as ha

# Camera configuration
DEVICE_ID = '00d0243dd49c_CognexCorporation_DS1300'
SAVE_DIR = './captures'
os.makedirs(SAVE_DIR, exist_ok=True)

def init_camera():
    acq_handle = ha.open_framegrabber(
        'GigEVision2',
        1, 1, 0, 0, 0, 0,
        'progressive', -1, 'default', -1, 'false', 'default',
        DEVICE_ID, 0, -1
    )
    ha.set_framegrabber_param(acq_handle, 'TriggerMode', 'On')
    ha.set_framegrabber_param(acq_handle, 'grab_timeout', -1)
    ha.grab_image_start(acq_handle, -1)
    return acq_handle

def capture_loop(acq_handle):
    image_counter = 0
    print("📸 Waiting for trigger...")

    try:
        while True:
            image = ha.grab_image_async(acq_handle, -1)
            image_path = os.path.join(SAVE_DIR, f"laser_trigger_{image_counter:03d}.png")
            plot_path = os.path.join(SAVE_DIR, f"tread_plot_{image_counter:03d}.png")

            ha.write_image(image, 'png', 0, image_path)
            print(f"✅ Saved image: {image_path}")

            analyze_and_plot_tread_profile(image_path, plot_output_path=plot_path)
            image_counter += 1

            # time.sleep(0.2)  # slight pause to avoid flood in case of rapid triggers
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")

def main():
    acq_handle = init_camera()
    try:
        capture_loop(acq_handle)
    finally:
        ha.close_framegrabber(acq_handle)

if __name__ == "__main__":
    main()
