"""
PSO (Particle Swarm Optimization) Algorithm
Opt Test Functions
"""

import autograd.numpy as np

# Defining an optimization test function
def Rastrigin(X,N):
    y = 10 * N + np.sum([(x**2 - 10 * np.cos(2 * np.pi * x)) for x in X])
    return y

# global minimum : f(0,0)=0

def Sphere(X,N):
    y = np.sum((x - 1)**2 for x in X) - 4
    return y

# global minimum : f(1,1)=-4