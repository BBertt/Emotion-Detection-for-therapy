from deepface import DeepFace
import os

# img1_path = '../Emotion_Recognition/tests/dataset/img1.jpg'
# img3_path = '../Emotion_Recognition/tests/dataset/img3.jpg'

# img_path = './captured_images/8.jpg'

# resp = DeepFace.verify(img1_path=img1_path, img2_path=img3_path)

import os
from deepface import DeepFace

count = 1
while True:
    path = f"./captured_images/{count}.jpg"

    if os.path.exists(path):
        try:
            obj = DeepFace.analyze(img_path=path, actions='emotion')
            result = obj[0]
            emotion = result['dominant_emotion']
            print(emotion)
        except ValueError as e:
            print(f"Error processing {path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {path}: {e}")
    else:
        break

    count += 1
