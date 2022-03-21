import numpy as np
from numpy import *
import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib import cm 
import seaborn as sb

def func(x, alpha=2.5):
    return alpha*x**2

def derivative(x,y):
    yprime = diff(y)/diff(x)
    yprime = insert(yprime, 0, 0, axis=None)
    xprime = array([(x[i+1] + x[i])/2 for i in range(len(yprime)-1)])
    xprime = insert(xprime, 0, 0, axis=None)
    return yprime, xprime 

x = linspace(0,100)
y = func(x)
y_prime, delta_x = derivative(x,y)
x,y = meshgrid(x,y)

f = lambda y_prime, delta_x: -cumsum(2*np.pi*y*np.sqrt(1 + y_prime**2)*delta_x)
g = lambda y_prime, delta_x: cumsum(2*np.pi*y*np.sqrt(1 + y_prime**2)*delta_x)
F = f(y_prime,delta_x)
F = F.reshape((50,50))
G = g(y_prime,delta_x)
G = G.reshape((50,50))
fig = plt.figure(figsize = [12,8])
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(y, x, F, cmap=cm.coolwarm)
ax.plot_surface(y, x, G, cmap=cm.coolwarm)
#plt.scatter(Z1,z3,z2,c='black', marker='o', alpha=0.9)
ax.set_xlabel('Sustainability')
ax.set_ylabel('Production')
ax.set_zlabel('Cost')
plt.show()