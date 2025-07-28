# Import the necessary libraries
import serial
import numpy as np
import cv2
import time
import h5py
import os
from datetime import datetime

def get_user_input():
    """Get capture duration from user"""
    while True:
        try:
            duration = float(input("Enter capture duration in seconds (0 for continuous): "))
            if duration < 0:
                print("Please enter a non-negative number.")
                continue
            return duration
        except ValueError:
            print("Please enter a valid number.")

def create_hdf5_file():
    """Create HDF5 file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"thermal_capture_{timestamp}.h5"
    return filename

def save_to_hdf5(filename, images, metadata):
    """Save images and metadata to HDF5 file"""
    with h5py.File(filename, 'w') as f:
        # Create datasets
        img_dataset = f.create_dataset('thermal_images', data=np.array(images), compression='gzip')
        
        # Save metadata
        meta_group = f.create_group('metadata')
        for key, value in metadata.items():
            if isinstance(value, str):
                meta_group.attrs[key] = value
            else:
                meta_group.attrs[key] = value
        
        # Save individual frame timestamps if available
        if 'frame_timestamps' in metadata:
            f.create_dataset('frame_timestamps', data=metadata['frame_timestamps'], compression='gzip')
    
    print(f"Data saved to {filename}")

# Serial setup
ser = serial.Serial('/dev/ttyUSB0', 921600)    # Change to COM port on Windows

SYNC_MARKER = bytes([0xAA, 0xBB])

# Get user input for capture duration
capture_duration = get_user_input()
continuous_mode = capture_duration == 0

# Initialize variables for data collection
captured_images = []
frame_timestamps = []
fps_values = []

# Fps calculation variables
last_marker_time = time.time()
fps = 0

# Timing variables
start_time = time.time()
frame_count = 0

# Create HDF5 filename
hdf5_filename = create_hdf5_file()

print(f"Starting capture...")
if continuous_mode:
    print("Continuous mode - Press 'ESC' to stop and save")
else:
    print(f"Capturing for {capture_duration} seconds...")

try:
    while True:
        # Check if we should stop (duration-based or continuous mode)
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if not continuous_mode and elapsed_time >= capture_duration:
            print(f"Capture duration reached ({capture_duration}s)")
            break
        
        # Wait for sync marker
        marker = ser.read(2)
        if marker == SYNC_MARKER:
            current_time = time.time()
            time_since_last_marker = current_time - last_marker_time
            fps = 1.0 / time_since_last_marker
            last_marker_time = current_time
            print(f"FPS: {fps:.2f} | Frame: {frame_count + 1} | Elapsed: {elapsed_time:.1f}s")

            data = ser.read(3072)  # 768 floats (32-bit)
            if len(data) == 3072:
                # Convert to 24x32 float array
                thermal_data = np.frombuffer(data, dtype=np.float32).reshape((24, 32))
                thermal_data_calibrated = 0.693 * thermal_data + 1.52

                # Store raw thermal data and metadata
                captured_images.append(thermal_data_calibrated.copy())
                frame_timestamps.append(current_time)
                fps_values.append(fps)
                frame_count += 1
                
                # Normalize to 8-bit (0-255) for OpenCV display
                # normalized = cv2.normalize(thermal_data_calibrated, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                normalized = cv2.normalize(thermal_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Rotate image by 180 degrees
                normalized = cv2.rotate(normalized, cv2.ROTATE_180)
                # Upscale for better visualization
                enlarged = cv2.resize(normalized, (320, 240), interpolation=cv2.INTER_LINEAR)
                
                # Apply color map
                colored = cv2.applyColorMap(enlarged, cv2.COLORMAP_INFERNO)
                
                # Add info overlay
                info_text = f"Frame: {frame_count} | FPS: {fps:.1f} | Time: {elapsed_time:.1f}s"
                cv2.putText(colored, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display
                cv2.imshow('Thermal Camera', colored)

        # Exit on 'ESC' key
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for 'ESC'
            print("ESC pressed - stopping capture")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Close serial and OpenCV
    ser.close()
    cv2.destroyAllWindows()
    
    # Save data if we captured any frames
    if captured_images:
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Prepare metadata
        metadata = {
            'capture_start_time': datetime.fromtimestamp(start_time).isoformat(),
            'capture_end_time': datetime.fromtimestamp(end_time).isoformat(),
            'total_duration_seconds': total_duration,
            'total_frames': len(captured_images),
            'average_fps': len(captured_images) / total_duration if total_duration > 0 else 0,
            'min_fps': min(fps_values) if fps_values else 0,
            'max_fps': max(fps_values) if fps_values else 0,
            'image_shape': f"{captured_images[0].shape}",
            'data_type': str(captured_images[0].dtype),
            'serial_port': 'COM3',
            'baud_rate': 921600,
            'frame_timestamps': np.array(frame_timestamps)
        }
        
        # Save to HDF5
        print(f"\nSaving {len(captured_images)} frames to HDF5...")
        save_to_hdf5(hdf5_filename, captured_images, metadata)
        
        # Print summary
        print(f"\nCapture Summary:")
        print(f"- Duration: {total_duration:.2f} seconds")
        print(f"- Frames captured: {len(captured_images)}")
        print(f"- Average FPS: {metadata['average_fps']:.2f}")
        print(f"- File saved: {hdf5_filename}")
        print(f"- File size: {os.path.getsize(hdf5_filename) / (1024*1024):.2f} MB")
    else:
        print("No frames captured")