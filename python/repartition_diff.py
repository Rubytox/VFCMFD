#!/usr/bin/env python

import sys
import numpy as np


def main():
    clusters = dict()
    outliers = dict()
    comp = dict()
    with open(sys.argv[1], "r") as data:
        line = data.readline()
        while line and 'number' not in line:
            nature, idx1, idx2, angle, norm = line.split(",")
            idx1 = int(idx1)
            idx2 = int(idx2)
            idx1, idx2 = (idx2, idx1) if idx2 < idx1 else (idx1, idx2)
            norm = abs(float(norm))
            angle = float(angle)

            if angle >= 180:
                angle -= 360
            elif angle <= -180:
                angle += 360

            angle = abs(angle)

            if nature == "cluster":
                clusters[(idx1, idx2)] = (angle, norm)
            else:
                outliers[(idx1, idx2)] = (angle, norm)

            line = data.readline()

    with open(sys.argv[2], "r") as data:
        line = data.readline()
        while line and 'number' not in line:
            idx1, idx2, angle, norm = line.split(",")
            idx1 = int(idx1)
            idx2 = int(idx2)
            idx1, idx2 = (idx2, idx1) if idx2 < idx1 else (idx1, idx2)
            norm = abs(float(norm))
            angle = float(angle)

            if angle >= 180:
                angle -= 360
            elif angle <= -180:
                angle += 360

            angle = abs(angle)

            comp[(idx1, idx2)] = (angle, norm)

            line = data.readline()

    with open(sys.argv[1] + ".res", "w") as out:
        for match in comp:
            if match in clusters:  # Correct
                out.write(F"c,{comp[match][0]},{comp[match][1]}\n")
            elif match in outliers:  # Not a match
                out.write(F"o,{comp[match][0]},{comp[match][1]}\n")
            else:  # Wrong or new
                idx1 = match[0]
                idx2 = match[1]
                genK1 = list(key[0] for key in clusters)
                genK2 = list(key[1] for key in clusters)
                genO1 = list(key[0] for key in outliers)
                genO2 = list(key[1] for key in outliers)
                if (idx1 in genK1 or idx2 in genK2
                        or idx1 in genO1 or idx2 in genO2):
                    out.write(F"w,{comp[match][0]},{comp[match][1]}\n")
                else:  # New
                    out.write(F"n,{comp[match][0]},{comp[match][1]}\n")

    # plt.title("Répartition des différences des normes en fonction des différences des angles")
    # ax = plt.gca()
    # ax.set_xlabel("Différences des angles")
    # ax.set_ylabel("Différences des normes")
    # plt.show()


if __name__ == "__main__":
    main()
