import numpy as np
import tensorflow as tf
import json
from PIL import Image
image_size = 224
def process_image(image):
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /=255
    return image.numpy()
def load_map(path_map):
    with open(path_map, 'r') as f:
        class_names = json.load(f)
def predict_image(image_path, model, top_k, path_map):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    probs_predict = model.predict(np.expand_dims(processed_test_image, axis=0))
    probs_ascend = np.sort(probs_predict)
    top_k_probs = probs_ascend.squeeze()[::-1][:top_k].tolist()
    top_k_class_indices = probs_predict.argsort().squeeze()[::-1][:top_k]
    top_k_classes = [str(i+1) for i in top_k_class_indices]
    class_names = load_map(path_map)
    top_k_names = [class_names[label] for label in top_k_classes]
    return top_k_probs , top_k_classes, top_k_names
