import os
import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")

count = 0
while True:
    path = f"./captured_images/{count}.jpg"

    if os.path.exists(path):
        try:
            img = cv2.imread(path)
            if img is not None:
                # Convert the image to grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Resize the image and expand dimensions as needed for the model
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_img, (48, 48)), -1), 0)
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                print(emotion_dict[maxindex])
        except ValueError as e:
            print(f"Error processing {path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {path}: {e}")
    else:
        break

    count += 1