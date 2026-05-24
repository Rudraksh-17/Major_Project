from keras.models import load_model
import cv2
import numpy as np

# Load CNN model
model = load_model("../model/drowsiness_cnn.h5")


def detect_drowsiness(image_path):

    # Read image
    img = cv2.imread(image_path)

    # Resize image
    img = cv2.resize(img, (64, 64))

    # Normalize
    img = img / 255.0

    # Reshape
    img = np.reshape(img, (1, 64, 64, 3))

    # Predict
    prediction = model.predict(img)[0][0]

    return float(prediction)