import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model with a path relative to this file to avoid CWD issues
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "asl_alphabet_cnn.h5")
model = load_model(MODEL_PATH)

# Classes must match training order produced by flow_from_directory (A..Z, del, nothing, space)
classes = [chr(i) for i in range(65, 91)] + ["del", "nothing", "space"]  # 29 classes


def predict_single_image(image_path):
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    label_index = int(np.argmax(prediction))
    label = classes[label_index]
    return label
