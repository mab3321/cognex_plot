import serial
import os
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600

def read_voltage(acq_handle):
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    was_below_threshold = False
    image_counter = 0

    print("📡 Monitoring voltage...")
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                try:
                    voltage = float(line)
                    print(f"Current voltage: {voltage}V")
                    # if voltage < THRESHOLD:
                    #     if not was_below_threshold:
                    #         print(f"⚠️ Voltage dropped: {voltage}V")
                    #         image_counter = capture_image(acq_handle, image_counter)
                    #         was_below_threshold = True
                    # else:
                    #     was_below_threshold = False
                except ValueError:
                    print("⚠️ Invalid voltage input.")
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        ser.close()
read_voltage(0)