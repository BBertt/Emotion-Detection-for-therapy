import pandas as pd
from sklearn.metrics import confusion_matrix

# Load the CSV file
data = pd.read_csv('Emotion_Recognition_Accuracy.csv')

# Assuming 'actual_labels' and 'predicted_labels' are the column names in your CSV file
actual_labels = data['ground_truth']
predicted_labels = data['predicted']

# Compute confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)

# Extract TP, FP, FN from the confusion matrix
TP = 0
FP = 0
FN = 0

positive_classes = ["Neutral", "Angry", "Disgusted", "Fearful", "Happy", "Sad"]

for actual, predicted in zip(actual_labels, predicted_labels):
    if actual == predicted and actual in positive_classes:  # Both actual and predicted are positive classes
        TP += 1
    elif predicted in positive_classes and actual not in positive_classes:  # Predicted positive, but actual is negative
        FP += 1
    elif actual in positive_classes and actual != predicted:  # Actual positive, but predicted is negative
        FN += 1

# Calculate precision
precision = TP / (TP + FP) if (TP + FP) != 0 else 0

# Calculate recall
recall = TP / (TP + FN) if (TP + FN) != 0 else 0

# Calculate F1 score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
