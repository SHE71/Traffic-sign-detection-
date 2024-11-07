import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

#Assigning Path for Dataset
data_dir = r"D:\project\TRAFFIC_sign\archive (1)"
train_path = r'D:\project\TRAFFIC_sign\archive (1)\Train'
test_path = r'D:\project\TRAFFIC_sign\archive (1)\Test'
test_data = r'D:\project\TRAFFIC_sign\archive (1)\Test.csv'

# Load the model from the .h5 file
model = tf.keras.models.load_model('D:\project\TRAFFIC_sign\model.h5')

# Summary of the model (optional, to check if it's loaded correctly)
model.summary()


# Loading the test data
test = pd.read_csv(test_data)  # Ensure `test_data` is defined with the path to the CSV file

labels = test["ClassId"].values
imgs = test["Path"].values

data = []

for img in imgs:
    try:
        image = cv2.imread(data_dir + '/' + img)  # Ensure `data_dir` is defined with the correct path
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((30, 30))
        data.append(np.array(resize_image))
    except Exception as e:
        print(f"Error in {img}: {e}")

X_test = np.array(data)
X_test = X_test / 255.0

# Running predictions on the test data
predictions = model.predict(X_test)
pred = np.argmax(predictions, axis=-1)

# Accuracy with the test data
print('Test Data accuracy:', accuracy_score(labels, pred) * 100)

# Assuming 'classes' is a list of class names or labels, e.g., classes = ["Class1", "Class2", ...]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels, pred)

# Convert confusion matrix to a DataFrame for easier plotting
df_cm = pd.DataFrame(cf)

# Plot the confusion matrix
plt.figure(figsize=(20, 20))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")
plt.show()

#Classification report
from sklearn.metrics import classification_report

print(classification_report(labels, pred))

#Predictions on Test Data
plt.figure(figsize = (25, 25))

start_index = 0
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = pred[start_index + i]
    actual = labels[start_index + i]
    col = 'g'
    if prediction != actual:
        col = 'r'
    plt.xlabel('Actual={} || Pred={}'.format(actual, prediction), color = col)
    plt.imshow(X_test[start_index + i])
plt.show()
