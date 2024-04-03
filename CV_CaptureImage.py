import cv2
import time
import os

def capture_images(interval, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 for default camera, change if you have multiple cameras

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    # Variable to keep track of image count
    count = 0

    # Infinite loop to capture images
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Show the captured frame
            cv2.imshow('Camera Feed', frame)

            # Save the captured frame as an image
            image_path = os.path.join(output_folder, f"{count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")

            # Increment image count
            count += 1

            # Wait for the specified interval (in seconds)
            time.sleep(interval)
            
            # Press 'q' to exit the loop and close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to capture frame.")
            break

    # Release the camera when finished
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set the interval for capturing images (in seconds)
    interval = 1

    # Set the folder where images will be saved
    output_folder = "captured_images"

    # Call the function to capture images
    capture_images(interval, output_folder)
