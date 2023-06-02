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

occurrences_left = [0] * 256
h = noise.shape[0]
w = noise.shape[1]
nb_left = 0
occurrences_right = [0] * 256
nb_right = 0
for y in range(h):
    for x in range(w):
        if mask_left[y, x] == 0xFF:
            occurrences_left[noise[y, x]] += 1
            nb_left += 1
        if mask_right[y, x] == 0xFF:
            occurrences_right[noise[y, x]] += 1
            nb_right += 1

for idx, _ in enumerate(occurrences_left):
    occurrences_left[idx] /= nb_left

for idx, _ in enumerate(occurrences_right):
    occurrences_right[idx] /= nb_right


X = list(range(0, 256))
ax = plt.gca()
ax.set_ylim([0, 0.14])
plt.plot(X, occurrences_left, c='green')
plt.plot(X, occurrences_right, c='red')
plt.show()
