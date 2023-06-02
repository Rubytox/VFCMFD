#!/usr/bin/env python

import math as m
import matplotlib.pyplot as plt

X = []  # Nb pixels image
N = []  # Nb points d'intérêt
# T en log de secondes !
T_kn = []  # Temps d'exécution en O(kn)
T_n2 = []  # Temps d'exécution en O(n^2)

# GRIP
X.append(786432)
N.append(10669)
T_kn.append(2.8)
T_n2.append(30)

# Bricks
X.append(10077696)
N.append(137606)
T_kn.append(174)
T_n2.append(5134)

# Paris
X.append(12052992)
N.append(161201)
T_kn.append(283)
T_n2.append(6987)

# DGA moyenne
X.append(13666824)
N.append(269277)
T_kn.append(837)
T_n2.append(36000)


ax1 = plt.gca()
ax1.set_xlabel('Number of pixels')
ax1.set_ylabel('Execution time (s)')

n2_line, = ax1.plot(X, T_n2, label="Regular g2NN time", color='orange')
kn_line, = ax1.plot(X, T_kn, label="Proposed time", color='green')

n2_legend = ax1.legend(handles=[n2_line], loc=(0.02, 0.9))
kn_legend = ax1.legend(handles=[kn_line], loc=(0.02, 0.85))

ax1.add_artist(n2_legend)
ax1.add_artist(kn_legend)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel("Number of keypoints", color=color)
N_line, = ax2.plot(X, N, label="Interest points", marker="2", color=color)
ax2.tick_params(axis='y', labelcolor=color)
N_legend = ax2.legend(handles=[N_line], loc=(0.02, 0.95))
ax2.add_artist(N_legend)

plt.draw()

locs = list(ax1.get_xticks())
labels = [w.get_text() for w in ax1.get_xticklabels()]
locs += [786432, 10077696, 12052992, 13666824]
labels += [r"GRIP = 786'432", r"Bricks = 10'077'696", r"Paris = 12'052'992", r"DGA = 13'666'824"]

plt.xticks(locs, labels)
ax1.grid()

plt.show()
