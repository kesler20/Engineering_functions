import math
#from typing_extensions import Concatenate
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import scipy
from scipy.integrate import odeint
from matplotlib import cm

class IdealReactor(object):
    def __init__(self,k_c=1,C_a=1,n=1,k_p=1,p_a=1,T=1,x_a=1,p_tot=1):
        self.k_c = k_c#rate_constant
        self.k_p = k_p#rate_constant
        self.C_a = C_a#concentration of A, C_a = (x_a*p_tot)/(RT)
        self.n = n#stoichiometry of A
        self.p_a = p_a#partial pressure of a as concentration for gasses
        self.p_tot = p_tot
        self.T = T
        self.x_a = x_a# x_a = p_a/p_tot
    
    def reaction(self,C,t):
        Ca = C[0]
        Cb = C[1]
        k = 2.0
        dAdt = -k*Ca
        dBdt = k*Ca
        return [dAdt, dBdt]

    def show_reaction_kinetics(self,C0, reaction, time):
        t = np.linspace(0,time)
        C = odeint(reaction,C0,t)
        plt.legend('Ca','Cb')
        plt.plot(t,C[:,0],'r--')
        plt.plot(t,C[:,1],'b-')
        plt.show()
    
    def rate_of_reaction(self,k_c,k_p,C_a,n,T,p_a):
        R = 8.314
        r_a = k_c*C_a**n
        r_a = k_p*p_a**n
        k_p = k_c/((R*T)**n)
        return r_a

    C_a0 = 2000
    k_c = 1.2
    t = []
    concentration = []
    y_plot = []
    for time in range(200):
        x = eval('math.log(C_a0, math.e) - k_c*time')
        C_a = C_a0 - math.e**(x)
        t.append(time)
        concentration.append(C_a)
        y_plot.append(x)

    plt.plot(t, y_plot)
    plt.show()

class batch(IdealReactor):
    def __init__(self,Na_0, concentration, C_a0, V):
        self.Na_0 = Na_0
        self.concentration = concentration
        self.C_a0 = C_a0
        self.V = V
        
    def conversion_number_of_moles(self,concentration,C_a0,V,Na_0):
        x_a = []
        for value in concentration:
            x = value*V/(1 - C_a0*V)
            x_a.append(x)
        Na = []
        for i in x_a:
            N = Na_0*(1 - i)
            Na.append(N)
        return Na, x_a


    temperature = [350, 400, 450, 500]
    k = [0.000398, 0.0141, 0.186, 1.48]
    K = []
    for value in k:
        cx = math.log(value, math.e)*-1
        K.append(cx)

    one_t = []
    for temper in temperature:
        T = 1/temper
        one_t.append(T)

    gradient = (K[0] - K[3])/(one_t[0]-one_t[3])
    delta_H = gradient*8.314
    print(delta_H)
    plt.plot(one_t, K)
    plt.show()

def three_d_plots(c1,c2,e1,e2, x_labels='a',y_labels='b', z_labels='c',linear_combination=True,c=0):
    if linear_combination:
        f = lambda x, y: c1*x**e1 + c2*y**e2
    else:
        f = lambda x, y: c*(x**(e1))*(y**(e2))
    x0 = np.linspace(0, 1)
    y0 = np.linspace(0, 1)
    X, Y = np.meshgrid(x0,y0)
    F = f(X,Y)

    fig = plt.figure(figsize = [12,8])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, F, cmap=cm.coolwarm)
    ax.set_xlabel(x_labels)
    ax.set_ylabel(y_labels)
    ax.set_zlabel(z_labels)
    plt.show()

#Concentration of [A,B]
reactor = IdealReactor(1,1,1,1,1,1,1,1)
C0 = [1,0]
reactor.show_reaction_kinetics(C0, reactor.reaction, 10)
