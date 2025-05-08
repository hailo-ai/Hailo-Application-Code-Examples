import numpy as np
import cv2
import argparse
import os
import sys
import signal

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nExiting...")
    cv2.destroyAllWindows()
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def convert_raw_image(input_file, width, height, format_type, save_output=False):
    """
    Convert raw image file to displayable format.
    
    Args:
        input_file (str): Path to the raw image file
        width (int): Width of the image
        height (int): Height of the image
        format_type (str): Either 'rgb' or 'nv12'
        save_output (bool): Whether to save the output as PNG
    """
    try:
        if not os.path.exists(input_file):
            print(f"Error: File {input_file} does not exist")
            return

        # Read raw file
        with open(input_file, "rb") as f:
            raw = f.read()

        if format_type == 'nv12':
            # NV12 format: Y plane (full) + UV plane (half)
            frame_size = int(width * height * 1.5)
            if len(raw) != frame_size:
                print(f"Error: Expected NV12 size {frame_size}, got {len(raw)}")
                return

            # Convert raw to numpy array and reshape to NV12 layout
            nv12 = np.frombuffer(raw, dtype=np.uint8)
            yuv = nv12.reshape((int(height * 3 / 2), width))
            
            # Convert NV12 to BGR
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        
        elif format_type == 'rgb':
            # RGB format: 3 bytes per pixel
            frame_size = width * height * 3
            if len(raw) != frame_size:
                print(f"Error: Expected RGB size {frame_size}, got {len(raw)}")
                return

            # Convert raw to numpy array and reshape to RGB layout
            rgb = np.frombuffer(raw, dtype=np.uint8)
            rgb = rgb.reshape((height, width, 3))
            
            # Convert RGB to BGR (OpenCV uses BGR)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        else:
            print(f"Error: Unsupported format {format_type}")
            return

        # Display the image
        window_name = f"{os.path.basename(input_file)} ({format_type.upper()})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, bgr)
        print(f"Displaying {input_file} ({format_type.upper()})")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # Give OpenCV time to process the destroy command

        # Optionally save the image
        if save_output:
            output_file = os.path.splitext(input_file)[0] + "_converted.png"
            cv2.imwrite(output_file, bgr)
            print(f"Saved converted image to {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Give OpenCV time to process the destroy command

def main():
    parser = argparse.ArgumentParser(description='Convert and display raw image files')
    parser.add_argument('input_file', help='Path to the raw image file')
    parser.add_argument('--width', type=int, required=True, help='Width of the image')
    parser.add_argument('--height', type=int, required=True, help='Height of the image')
    parser.add_argument('--format', choices=['rgb', 'nv12'], required=True, help='Format of the raw image (rgb or nv12)')
    parser.add_argument('--save', action='store_true', help='Save the converted image as PNG')
    
    args = parser.parse_args()
    convert_raw_image(args.input_file, args.width, args.height, args.format, args.save)

if __name__ == "__main__":
    main()



