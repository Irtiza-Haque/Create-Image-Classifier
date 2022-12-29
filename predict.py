import sys
import time 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import argparse
import json
from func_utils import predict_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classifier')
    parser.add_argument('image_path')
    parser.add_argument('irtiza.h5')
    parser.add_argument('--top_k', dest="top_k", type=int)
    parser.add_argument('--category_names', dest="category_names")
    args = parser.parse_args()
    
    model = tf.keras.models.load_model('irtiza.h5',custom_objects={'KerasLayer':hub.KerasLayer},compile=False)
    
    
    
    top_k_prob , top_k_classes, tok_k_names = predict_image(args.image_path, 
                                                       args.model, 
                                                            args.top_k, 
                                                            args.category_names)
    print(top_k_probs)
    print(top_k_classes)
    print(top_k_names)