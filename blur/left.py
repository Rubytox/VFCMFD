#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
mask_left = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
mask_right = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)

blur = cv2.blur(img, (5, 5))
noise = img - blur
noise += 128

occurrences = [0] * 256
h = noise.shape[0]
w = noise.shape[1]
for y in range(h):
    for x in range(w):
        occurrences[noise[y, x]] += 1

occurrences /= img.shape[0] * img.shape[1]

img_left = img.copy()
for y in range(h):
    for x in range(w):
        if mask_left[y, x] != 0:
            img_left[y, x] = 0
blur_left = cv2.blur(img_left, (5, 5))
noise_left = img_left - blur_left
noise_left += 128

occurrences_left = [0] * 256
for y in range(h):
    for x in range(w):
        occurrences_left[noise_left[y, x]] += 1

img_right = img.copy()
for y in range(h):
    for x in range(w):
        if mask_right[y, x] != 0:
            img_right[y, x] = 0
blur_right = cv2.blur(img_right, (5, 5))
noise_right = img_right - blur_right
noise_right += 128

occurrences_right = [0] * 256
for y in range(h):
    for x in range(w):
        occurrences_right[noise_right[y, x]] += 1

X = list(range(0, 256))
plt.plot(X, occurrences, c='blue')
plt.plot(X, occurrences_left, c='green')
plt.plot(X, occurrences_right, c='red')
plt.show()
