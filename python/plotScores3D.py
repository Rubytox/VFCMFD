#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt


def main():
    data = dict()
    with open(sys.argv[1], "r") as datafile:
        line = datafile.readline()
        while line:
            size, sensitivity, specificity = line.split(',')
            size = int(size)
            sensitivity = float(sensitivity)
            specificity = 1 - float(specificity)

            if size not in data:
                data[size] = [(0,0)]
            data[size].append((specificity, sensitivity))

            line = datafile.readline()
    for size in data:
        data[size].append((1, 1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for size in data:
        Sizes = []
        Sens = []
        Spec = []
        for spec, sens in data[size]:
            Sizes.append(size)
            Sens.append(sens)
            Spec.append(spec)
        ax.plot(Sizes, Spec, Sens)

    ax.set_xlabel("size")
    ax.set_ylabel("sensitivity")
    ax.set_zlabel("specificity")
    plt.show()


if __name__ == "__main__":
    main()
