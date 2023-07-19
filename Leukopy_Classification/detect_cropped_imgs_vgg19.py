from numpy.lib.shape_base import column_stack
#import streamlit as st
import matplotlib.pyplot as plt

from importlib import reload
from pathlib import Path
from PIL import Image
import os

import vgg19_utils as vgg19_utils
#import vgg16_utils as vgg16_utils
#import vit_b16_utils as vit_utils
import common, ui

reload(vgg19_utils)
#reload(vgg16_utils)
#reload(vit_utils)
#reload(common)

model_list = ["VGG16+SVM", "VGG19", "ViT-b16"]

#img_name = "A"
#def write():
    #global img_name

img_file = "test_image.jpg"
#img = img_file.read()


#img_name = img_file.name
#img_info = Image.open(img_file)
#file_details = f"""
#Name: {img_name}
#Type: {img_info.format}
#Size: {img_info.size}
#"""


#model_choice == "VGG19":
# Importe le modèle (en cache)
model = vgg19_utils.load_model()
# Prédiction + Grad-CAM

for img_file in os.listdir("test_detect"):
    fig, sorted_classes, sorted_preds = vgg19_utils.vgg19_prediction(
        model, "test_detect/"+img_file)

    # Prints the classes and their probability, ranked from highest to lowest.
    print(sorted_classes[0], vgg19_utils.print_proba(sorted_preds[0]))
    print(sorted_classes[1], vgg19_utils.print_proba(sorted_preds[1]))
    print(sorted_classes[2], vgg19_utils.print_proba(sorted_preds[2]))

    f = open("test_WBC_labels.txt", "a")
    f.write(img_file + " " + sorted_classes[0] + " " + vgg19_utils.print_proba(sorted_preds[0]) + "\n")
    f.close()


#print('Grad-CAM for', sorted_classes[0])
#fig = plt.figure() 
#plt.show()