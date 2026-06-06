# tread_depth_analyzer.py

import os
import time
from extract_tread_from_cognex_disparity import analyze_and_plot_tread_profile

# Add HALCON DLL directory for Python 3.8+ on Windows
if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    halconroot = os.environ.get('HALCONROOT')
    if halconroot:
        halcon_bin = os.path.join(halconroot, 'bin', 'x64-win64')
        if os.path.exists(halcon_bin):
            os.add_dll_directory(halcon_bin)

import halcon as ha

# Camera configuration
DEVICE_ID = "00d0243dd49c_CognexCorporation_DS1300"
FORCE_IP_PARAMS = "force_ip=169.254.171.200/00:D0:24:3D:D4:9C/169.254.171.1/255.255.255.0"
SAVE_DIR = "./captures"
os.makedirs(SAVE_DIR, exist_ok=True)


def init_camera():
    """Initialize camera with hardware trigger mode and force IP settings"""
    acq_handle = ha.open_framegrabber(
        "GigEVision2",
        0,  # HorizontalResolution
        0,  # VerticalResolution
        0,  # ImageWidth
        0,  # ImageHeight
        0,  # StartRow
        0,  # StartColumn
        "progressive",
        -1,  # BitsPerChannel
        "default",  # ColorSpace
        FORCE_IP_PARAMS,  # Generic
        "false",  # ExternalTrigger
        "default",  # CameraType
        DEVICE_ID,
        0,  # Port
        -1,  # LineIn
    )

    # Set trigger mode to ON (hardware trigger)
    ha.set_framegrabber_param(acq_handle, "TriggerMode", "On")

    # Start asynchronous grabbing
    ha.grab_image_start(acq_handle, -1)

    return acq_handle


def capture_loop(acq_handle):
    """Continuously capture images when hardware trigger fires"""
    image_counter = 0
    last_capture_time = 0
    print("📸 Waiting for hardware trigger...")

    try:
        while True:
            try:
                # Wait for trigger (timeout in milliseconds)
                image = ha.grab_image_async(acq_handle, 1000)

                # Debounce: only save if 2 seconds passed since last capture
                current_time = time.time()
                if current_time - last_capture_time >= 2:
                    image_path = os.path.join(
                        SAVE_DIR, f"laser_trigger_{image_counter:03d}.png"
                    )
                    plot_path = os.path.join(
                        SAVE_DIR, f"tread_plot_{image_counter:03d}.png"
                    )

                    ha.write_image(image, "png", 0, image_path)
                    print(f"✅ Saved image: {image_path}")

                    # Analyze and plot tread profile
                    analyze_and_plot_tread_profile(
                        image_path, plot_output_path=plot_path
                    )

                    image_counter += 1
                    last_capture_time = current_time
                else:
                    print("⏳ Ignored trigger (within 2s debounce)...")

            except ha.HOperatorError as e:
                if "timeout" in str(e).lower():
                    # Timeout is expected when waiting for trigger
                    continue
                else:
                    print(f"⚠️ HALCON error: {e}")

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")


def main():
    """Main entry point"""
    print("🚀 Starting camera with hardware trigger mode...")
    acq_handle = init_camera()
    try:
        capture_loop(acq_handle)
    finally:
        ha.close_framegrabber(acq_handle)
        print("📷 Camera closed.")


if __name__ == "__main__":
    main()
