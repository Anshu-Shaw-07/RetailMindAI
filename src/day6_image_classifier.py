import json
import numpy as np
import tensorflow as tf
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "day6_product_cnn.keras"
LABEL_PATH = BASE_DIR / "models" / "day6_class_names.json"
IMG_SIZE = 128


class ProductImageClassifier:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABEL_PATH, "r", encoding="utf-8") as file:
            self.class_names = json.load(file)

    def predict(self, image_path):
        image = tf.keras.utils.load_img(
            image_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode="rgb"
        )
        image = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        probabilities = self.model.predict(image, verbose=0)[0]
        index = int(np.argmax(probabilities))

        return {
            "category": self.class_names[index],
            "confidence": round(float(probabilities[index]) * 100, 2)
        }
from src.image_classifier import ProductImageClassifier

classifier = ProductImageClassifier()
result = classifier.predict("data/test_product.jpg")
print(result)