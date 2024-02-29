"""
PSO (Particle Swarm Optimization) Algorithm
Functions
"""
from random import random

# Functions

# Function that updates the Pb: Personal Best (from t to t + 1)
def update_Pb(X, Pb, S, Function, N):
    for s in range(S):
        if Function(X[s,], N) < Function(Pb[s,], N):
            Pb[s,] = X[s,]
    return Pb

# Function that updates the Gb: Global Best (from t to t + 1)
def update_Gb(Gb, Pb, S, Function, N):
    for s in range(0, S):
        if Function(Pb[s,], N) < Function(Gb, N):
            Gb = Pb[s,]
    return Gb

# Function that updates the velocity of the particles (from t to t + 1)
def update_V(w, c1, c2, X, V, Pb, Gb):
    r1 = random()
    r2 = random()
    V_updated = w * V + c1 * r1 * (Pb - X) + c2 * r2 * (Gb - X)
    return V_updated

# Function that updates the position of X (the particles) (from t to t + 1)
def update_X(X, V_updated):
    X_updated = X + V_updated
    return X_updated

# Penalty function that resets the coordinates of particles that exceed
# the search space. This function is much more complicated in the case of
# problems with constraints
def Penalty(X, N, S, ub):
    for s in range(S):
        for n in range(N):
            if abs(X[s, n]) > ub:
                X[s, n] = (-ub) + random() * 2 * ub
    return X