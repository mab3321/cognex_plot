import time,os
halcon_bin_path = r"C:\Users\DELL\AppData\Local\Programs\MVTec\HALCON-24.11-Progress-Student-Edition\bin\x64-win64"
os.environ["PATH"] = halcon_bin_path + os.pathsep + os.environ["PATH"]

import halcon as ha

# === Settings ===
DEVICE_ID = '00d0243dd49c_CognexCorporation_DS1300'
NUM_MEASUREMENTS = 10

def init_camera():
    acq_handle = ha.open_framegrabber(
        'GigEVision2',
        1, 1, 0, 0, 0, 0,
        'progressive', -1, 'default', -1, 'false', 'default',
        DEVICE_ID, 0, -1
    )
    ha.set_framegrabber_param(acq_handle, 'TriggerMode', 'On')
    ha.set_framegrabber_param(acq_handle, 'grab_timeout', -1)
    ha.set_framegrabber_param(acq_handle, 'CPProfilesPerFrame', 1)

    ha.grab_image_start(acq_handle, -1)
    return acq_handle

def measure_frame_intervals(acq_handle):
    intervals = []

    print("🕒 Measuring inter-frame time... Trigger the sensor manually.")
    try:
        # Prime the first grab
        ha.grab_image_async(acq_handle, 5000)

        for i in range(NUM_MEASUREMENTS):
            t0 = time.perf_counter()
            ha.grab_image_async(acq_handle, 5000)
            t1 = time.perf_counter()
            delta = t1 - t0
            intervals.append(delta)
            print(f"Frame {i+1}: Δt = {delta:.6f} seconds")

    except ha.HOperatorError as e:
        print(f"⚠️ HALCON Error: {e}")

    return intervals

def main():
    acq_handle = init_camera()
    try:
        intervals = measure_frame_intervals(acq_handle)
        if intervals:
            avg = sum(intervals) / len(intervals)
            print(f"\n📊 Average inter-frame time over {len(intervals)} frames: {avg:.6f} seconds")
    finally:
        ha.close_framegrabber(acq_handle)

if __name__ == "__main__":
    main()
