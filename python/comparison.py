#!/usr/bin/env python

import sys


def main():
    reference = []
    with open(sys.argv[1], "r") as data:
        line = data.readline()
        while line and 'number' not in line:
            x, y, theta, l = line.split(",")
            x = float(x)
            y = float(y)
            theta = float(theta)
            l = float(l)
            reference.append((x, y, theta, l))

            line = data.readline()

    compared_to = []
    with open(sys.argv[2], "r") as data:
        line = data.readline()
        while line and 'number' not in line:
            x, y, theta, l = line.split(",")
            x = float(x)
            y = float(y)
            theta = float(theta)
            l = float(l)
            compared_to.append((x, y, theta, l))

            line = data.readline()

    correct = 0
    missing = 0
    for match in reference:
        reverse = (match[1], match[0])
        if match in compared_to or reverse in compared_to:
            correct += 1
        else:
            missing += 1

    false_matches = 0
    for match in compared_to:
        reverse = (match[1], match[0])
        if match not in reference and reverse not in reference:
            false_matches += 1

    per_correct = round(correct / len(reference) * 100, 2)
    per_missing = round(missing / len(reference) * 100, 2)
    per_false_matches = round(false_matches / len(compared_to) * 100, 2)
    print(F"Total matches in O(n^2): {len(reference)}")
    print(F"Total matches in O(kn): {len(compared_to)}")
    print(F"Correct matches: {correct} ({per_correct} %)")
    print(F"Missing matches: {missing} ({per_missing} %)")
    print(F"Number of matches in O(kn) that are not in O(n^2): {false_matches} ({per_false_matches} %)")


if __name__ == "__main__":
    main()
