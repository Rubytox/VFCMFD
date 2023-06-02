#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import numpy as np


def main():
    donnees = []
    with open(sys.argv[1], "r") as data:
        line = data.readline()
        while line:
            color, angle, norm = line.split(",")
            norm = abs(float(norm))
            angle = float(angle)

            if angle >= 180:
                angle -= 360
            elif angle <= -180:
                angle += 360

            angle = abs(angle)

            donnees.append((color, angle, norm))

            line = data.readline()

    for color, angle, norm in donnees:
        draw_col = "black"
        if color == "c":
            draw_col = "green"
        elif color == "w":
            draw_col = "red"
        elif color == "n":
            draw_col = "yellow"
        elif color == "o":
            draw_col = "blue"
        plt.scatter(angle, norm, color=draw_col)

    plt.title("Répartition des différences des normes en fonction des différences des angles")
    ax = plt.gca()
    ax.set_xlabel("Différences des angles")
    ax.set_ylabel("Différences des normes")
    plt.show()


if __name__ == "__main__":
    main()
