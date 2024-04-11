import os
import cv2
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_result = [0] * 7

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
                if (maxindex == 0):
                    emotion_result[0] += 1
                elif (maxindex == 1):
                    emotion_result[1] += 1
                elif (maxindex == 2):
                    emotion_result[2] += 1
                elif (maxindex == 3):
                    emotion_result[3] += 1
                elif (maxindex == 4):
                    emotion_result[4] += 1
                elif (maxindex == 5):
                    emotion_result[5] += 1
                elif (maxindex == 6):
                    emotion_result[6] += 1
                elif (maxindex == 7):
                    emotion_result[7] += 1
        except ValueError as e:
            print(f"Error processing {path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {path}: {e}")
    else:
        break

    count += 1

# Displaying the final results
x_values = range(len(emotion_result))
x_labels = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
plt.bar(x_values, emotion_result, tick_label=x_labels)

plt.title('Emotion Shown over a Session')
plt.xlabel('Emotions')
plt.ylabel('Amount')

plt.grid(True)

#Give an overview of the most amount of emotion
highest_value = emotion_result[0]
highest_emotion = []

for x in emotion_result:
    if x > highest_value:
        highest_value = x

for i in range(len(emotion_result)):
    if emotion_result[i] == highest_value:
        highest_emotion.append(emotion_dict[i])

for x in highest_emotion:
    if (x == "Angry"):
        print("""The client is mostly feeling angry.
Anger can often arise as a response to feeling threatened, whether physically, emotionally, or psychologically.\n 
It serves as a signal that one's boundaries have been crossed or their needs have not been met.\n 
Exploring the underlying triggers and understanding the source of anger can lead to healthier coping mechanisms and conflict resolution strategies.""")
    elif (x == "Disgusted"):
        print("""The client is mostly feeling disgusted.
Disgust is a protective emotion, evolved to keep us away from potentially harmful or contaminating stimuli. 
When a client experiences disgust, it may indicate a violation of their moral or aesthetic standards. 
By examining the specific triggers and underlying beliefs associated with disgust, therapy can help clients navigate their values and boundaries more effectively.""")
    elif (x == "Fearful"):
        print("""The client is mostly feeling fearful.
Fear is a primal emotion triggered by perceived threats or dangers. 
When clients experience fear, it may stem from past traumas, perceived vulnerabilities, or uncertainty about the future. 
More therapy sessions can provide a safe space for exploring and processing these fears, gradually building resilience and adaptive coping strategies.""")
    elif (x == "Happy"):
        print("""The client is mostly feeling happy.
Happiness reflects a state of contentment, joy, and fulfillment.
When clients experience happiness, it often signifies the satisfaction of needs, achievement of goals, or positive social connections.
Exploring sources of happiness and enhancing positive experiences can contribute to overall well-being and resilience.""")
    elif (x == "Neutral"):
        print("""The client is mostly feeling neutral.
A neutral emotional state suggests a lack of strong emotional arousal, often observed in routine or mundane situations. 
While neutrality can be a natural part of the emotional spectrum, persistent neutrality may indicate emotional numbness or disconnection. 
More therapy sessions can help clients explore underlying emotions and cultivate a greater awareness of their internal experiences.""")
    elif (x == "Sad"):
        print("""The client is mostly feeling sad.
Sadness is a common response to loss, disappointment, or unmet expectations. 
When clients experience sadness, it may signal a need for emotional support, validation, or self-compassion. 
More therapy sessions can provide a compassionate space for processing grief, exploring underlying beliefs, and fostering resilience in the face of adversity.""")
    elif (x == "Surprised"):
        print("""The client is mostly feeing surprised.
Surprise arises from unexpected events or stimuli, activating our attention and arousal systems. 
When clients experience surprise, it can signal a shift in their expectations or perception of reality. 
Exploring the meaning and implications of surprise can help clients adapt to change, embrace novelty, and cultivate flexibility in their thinking and behavior.
""")
plt.show()

