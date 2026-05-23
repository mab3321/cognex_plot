import serial
import os
import sys
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

# Settings
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600
THRESHOLD = 5.0  # New threshold as requested
DEVICE_ID = '00d0243dd49c_CognexCorporation_DS1300'
SAVE_DIR = './captures'

os.makedirs(SAVE_DIR, exist_ok=True)

def init_camera():
    return ha.open_framegrabber(
        'GigEVision2',
        1, 1, 0, 0, 0, 0,
        'progressive', -1, 'default', -1, 'false', 'default',
        DEVICE_ID, 0, -1
    )

def capture_image(acq_handle, count):
    try:
        image, _, _, _ = ha.grab_data_async(acq_handle, 5000)
        image_filename = os.path.join(SAVE_DIR, f"voltage_drop_{count:03d}.png")
        plot_filename = os.path.join(SAVE_DIR, f"tread_plot_{count:03d}.png")

        ha.write_image(image, 'png', 0, image_filename)
        print(f"✅ Captured and saved: {image_filename}")

        analyze_and_plot_tread_profile(image_filename, plot_output_path=plot_filename)

        return count + 1
    except ha.HOperatorError:
        print("⚠️ HALCON capture failed.")
        return count


def read_voltage(acq_handle):
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    last_voltage = THRESHOLD  # Assume it starts at or above threshold
    image_counter = 0

    print("📡 Monitoring voltage...")
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                try:
                    voltage = float(line)
                    # Trigger only if falling below threshold
                    if last_voltage >= THRESHOLD and voltage < THRESHOLD:
                        print(f"⚠️ Voltage dropped: {last_voltage}V ➞ {voltage}V")
                        image_counter = capture_image(acq_handle, image_counter)
                        print("⏳ Waiting 10 seconds before next read...")
                        time.sleep(10)  # Wait before checking again
                    last_voltage = voltage  # Update last voltage
                except ValueError:
                    print("⚠️ Invalid voltage input.")
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        ser.close()

def main():
    acq_handle = init_camera()
    try:
        read_voltage(acq_handle)
    finally:
        ha.close_framegrabber(acq_handle)

if __name__ == "__main__":
    main()
