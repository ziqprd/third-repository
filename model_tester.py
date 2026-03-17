import os
import tensorflow as tf
from PIL import Image
import cv2
from matplotlib import pyplot as plt

DATASET_PATH = "dataset/"
num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Ошибка, файл не найден по пути: {image_path}")
        return
    try:
        img = Image.open(image_path)
        img.verify()
    except (OSError, IOError):
        print(f"Ошибка, повреждённое изображение - {image_path}")
        return
    model = tf.keras.models.load_model("animal_image_classifier.h5")
    img = cv2.imread(image_path)
    displayimg = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка, не удалось прочитать изображение - {image_path}")
        return
    img = cv2.resize(img, (128, 128))
    img = img / 255
    img = tf.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_names = os.listdir(DATASET_PATH)
    if class_mode == "binary":
        predicted_class = class_names[int(bool(prediction[0] > 0.5))]
    else:
        predicted_class = class_names[tf.argmax(prediction, axis=-1).numpy()[0]]
    print(f"Модель опредилила: {predicted_class}")
    plt.imshow(cv2.cvtColor(displayimg, cv2.COLOR_BGR2RGB))
    plt.title(f'Модель определила: {predicted_class}')
    plt.axis("off")
    plt.show()

predict_image("dataset/parrots/angietoh-parrot-2909828_1280.jpg")










