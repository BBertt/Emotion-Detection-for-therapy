import os
import cv2
import time

def capture_images(interval, output_folder):

    # load json and create model
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

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

        if not ret:
            break

        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]

            # Save the captured frame as an image
            image_path = os.path.join(output_folder, f"{count}.jpg")
            cv2.imwrite(image_path, roi_gray_frame)
            print(f"Image saved: {image_path}")

            # Increment image count
            count += 1
       
        # Show the captured frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' to exit the loop and close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Wait for the specified interval (in seconds)
        time.sleep(interval)

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
