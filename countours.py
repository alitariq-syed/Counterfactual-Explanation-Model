# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:20:12 2021

@author: Ali
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('test.png')
plt.imshow(im), plt.show()

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[4]
cv2.drawContours(im, contours, -1, (255,0,0), 3)

#plt.imshow(thresh), plt.show()
plt.imshow(im), plt.show()
