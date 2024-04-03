from deepface import DeepFace

# img1_path = '../Emotion_Recognition/tests/dataset/img1.jpg'
# img3_path = '../Emotion_Recognition/tests/dataset/img3.jpg'

img_path = '../Emotion_Recognition/tests/dataset/img1.jpg'

# resp = DeepFace.verify(img1_path=img1_path, img2_path=img3_path)

obj = DeepFace.analyze(img_path= img_path, actions='emotion')

result = obj[0]

emotion = result['dominant_emotion']

print(emotion)