# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 19:34:21 2023

@author: Ali
"""

import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import load_img

figs = ["model_debugging_work/figures/New Misclassification/selected/26_Shiny_Cowbird_0031_796851.jpg",
        "model_debugging_work/figures/New Misclassification/selected/63_Ring_Billed_Gull_0021_51300.jpg",
        "model_debugging_work/figures/New Misclassification/selected/104_Whip_Poor_Will_0022_796438.jpg"]

for fig_path in figs:
    fig = load_img(fig_path, grayscale=False, color_mode="rgb", target_size=(224,224))
    fig_name = fig_path.split('/')[-1].split('.')[0]
    plt.imshow(fig), plt.axis('off'), plt.title(''),plt.savefig(fname="./model_debugging_work/figures/New Misclassification/selected/"+fig_name+".png", dpi=300, bbox_inches = 'tight'), plt.show()