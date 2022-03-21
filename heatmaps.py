import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sb 

def heat_plot(x0=0,x1=1,y0=0,y1=1,m_final=169):
    x = np.linspace(x0,x1)
    y = np.linspace(y0,y1)
    x, y = np.meshgrid(x,y)
    m = 1
    f = lambda x,y: (2*(1 - (-1)**m)/(m*np.pi*np.sinh(m*np.pi)))*np.sin(m*np.pi*x)*np.sinh(m*np.pi*y)
    F = f(x,y)
    # check why first response looks a lot better
    # then as you add up series artefacts start showing up
    df = pd.DataFrame(F)
    for m in range(2,m_final):
        f = lambda x,y: (2*(1 - (-1)**m)/(m*np.pi*np.sinh(m*np.pi)))*np.sin(m*np.pi*x)*np.sinh(m*np.pi*y)
        F = f(x,y)
        data = pd.DataFrame(F)
        for i in range(len(df)):
            df[i] += data[i]
        if m == 2:
            print(m)
            F = np.array(df)
            ax = sb.heatmap(F, cmap=cm.hot)
            ax.invert_yaxis()
            ax.plot()
            plt.show()
        elif m == 40:
            print(m)
            F = np.array(df)
            ax = sb.heatmap(F, cmap=cm.hot)
            ax.invert_yaxis()
            ax.plot()
            plt.show()
        elif m == 89:
            print(m)
            F = np.array(df)
            ax = sb.heatmap(F, cmap=cm.hot)
            ax.invert_yaxis()
            ax.plot()
            plt.show()
        elif m == 120:
            print(m)
            F = np.array(df)
            ax = sb.heatmap(F, cmap=cm.hot)
            ax.invert_yaxis()
            ax.plot()
            plt.show()
        elif m == 169:
            print(m)
            F = np.array(df)
            ax = sb.heatmap(F, cmap=cm.hot)
            ax.invert_yaxis()
            ax.plot()
            plt.show()
        else:
            pass

heat_plot()