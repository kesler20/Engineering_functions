import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
from matplotlib import cm
import seaborn as sb

def calculate_graph(x1, y1, x_0=0, y_0=0, beta0=0):
    x = np.linspace(x_0,x1)
    y = np.linspace(y_0, y1)
    z = lambda x,y : ((0.9918*y)**0.9)/(x**0.9)
    x, y = np.meshgrid(x,y)
    f = z(x,y)
    fig = plt.figure(figsize = [12,8])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, f, cmap=cm.coolwarm)
    #plt.scatter(Z1,z3,z2,c='black', marker='o', alpha=0.9)
    ax.set_xlabel('Sustainability')
    ax.set_ylabel('Production')
    ax.set_zlabel('Cost')
    plt.show()

