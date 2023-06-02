#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

cv2.imwrite("grayscale.jpg", img)

blur = cv2.blur(img, (5, 5))

cv2.imwrite("blured.jpg", blur)

noise = img - blur
noise += 128

occurrences = [0] * 256
h = noise.shape[0]
w = noise.shape[1]

for y in range(h):
    for x in range(w):
        occurrences[noise[y, x]] += 1

X = list(range(0, 256))
plt.plot(X, occurrences)
plt.show()
