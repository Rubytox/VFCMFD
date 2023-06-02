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
N.append(m.log(10669))
T_kn.append(m.log(2.8))
T_n2.append(m.log(30))

# Bricks
X.append(10077696)
N.append(m.log(137606))
T_kn.append(m.log(174))
T_n2.append(m.log(30))

# Paris
X.append(12052992)
N.append(m.log(161201))
T_kn.append(m.log(283))
T_n2.append(m.log(30))

# DGA moyenne
X.append(13666824)
N.append(m.log(269277))
T_kn.append(m.log(837))
T_n2.append(m.log(30))



N_line, = plt.plot(X, N, label="Interest points")
n2_line, = plt.plot(X, T_n2, label="O(n^2) time")
kn_line, = plt.plot(X, T_kn, label="O(kn) time")

N_legend = plt.legend(handles=[N_line], loc='upper left')
n2_legend = plt.legend(handles=[n2_line], loc='lower left')
kn_legend = plt.legend(handles=[kn_line], loc='upper right')

ax = plt.gca()
ax.add_artist(N_legend)
ax.add_artist(n2_legend)
ax.add_artist(kn_legend)


plt.show()
