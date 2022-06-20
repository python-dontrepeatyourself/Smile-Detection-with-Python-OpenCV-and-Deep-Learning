from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import os    
    
#######################################################################
## Loading the Data
#######################################################################

    
valid_formats = [".jpg", ".jpeg", ".png"]

def image_paths(root):
    "get the full path to each image in the dataset"
    image_paths = []

    # loop over the diretory tree
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            # extract the file extension from the filename
            extension = os.path.splitext(filename)[1].lower()

            # if the filename is an image we build the full 
            # path to the image and append it to our list
            if extension in valid_formats:
                image_path = os.path.join(dirpath, filename)
                image_paths.append(image_path)
    
    return image_paths
    
image_paths = image_paths("SMILES")
IMG_SIZE = [32, 32]

def load_dataset(image_paths, target_size=IMG_SIZE):
    data = []
    labels = []
    
    # loop over the image paths
    for image_path in image_paths:
        # load the image in grayscale and convert it to an array
        image = load_img(image_path, color_mode="grayscale", target_size=target_size)
        image = img_to_array(image)
        # append the array to our list 
        data.append(image)
        
        # extract the label from the image path and append it to the `labels` list
        label = image_path.split(os.path.sep)[-3]
        label = 1 if label == "positives" else 0
        labels.append(label)
        
    return np.array(data) / 255.0, np.array(labels)

data, labels = load_dataset(image_paths, IMG_SIZE)


##########################################################################
## Training the Smile Detector
##########################################################################


def build_model(input_shape=IMG_SIZE + [1]):
    model = Sequential([
        Conv2D(filters=32,
               kernel_size=(3, 3),
               activation="relu",
               padding='same',
               input_shape=input_shape),
        MaxPool2D(2, 2),
        Conv2D(filters=64,
               kernel_size=(3, 3),
               activation="relu",
               padding='same'),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# count the number of each label 
label, counts = np.unique(labels, return_counts=True)
# compute the class weights
counts = max(counts) / counts
class_weight = dict(zip(label, counts))


(X_train, X_test, y_train, y_test) = train_test_split(data, labels, 
                                                    test_size=0.2,
                                                    stratify=labels,
                                                    random_state=42)

(X_train, X_valid, y_train, y_valid) = train_test_split(X_train, y_train,
                                                        test_size=0.2,
                                                        stratify=y_train,
                                                        random_state=42)

# build the model
model = build_model()

# train the model
EPOCHS = 20
history = model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    class_weight=class_weight,
                    batch_size=64,
                    epochs=EPOCHS)

# save the model
model.save("model")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')

# plot the learning curves of the training and validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(range(EPOCHS), acc, "b", label="Training Accuracy")
plt.plot(range(EPOCHS), val_acc, "r", label="Validation Accuracy")
plt.legend()

plt.figure()

plt.plot(range(EPOCHS), loss, "g", label="Training Loss")
plt.plot(range(EPOCHS), val_loss, "orange", label="Validation Loss")
plt.legend()

plt.show()

