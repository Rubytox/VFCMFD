#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

img_orig = cv2.imread(sys.argv[1])
img_fals = cv2.imread(sys.argv[2])

h = img_orig.shape[0]
w = img_orig.shape[1]

mask = np.zeros((h, w))

for y in range(0, h):
    for x in range(0, w):
        if (img_orig[y, x] != img_fals[y, x]).all():
            mask[y, x] = 255

cv2.imwrite("skate_mask.jpg", mask)
