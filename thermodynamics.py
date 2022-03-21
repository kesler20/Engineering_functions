import math
import sympy
from sympy import Symbol, integrate


''' Equilibrium relationships assumptions

    Equilibrium relationships have the assumption that the species i in question is the 
    MVC more volatile component and the system is at equilibrium.

    we often operate at constant total pressure or constant temperature
'''
def Temperature(p_star_vap_a, A, B, C):
    T = B/(A - math.log(p_star_vap_a, math.e)) - C
    print(T)
    return int(T)

def Antoine_equation(A, B, C, T):
    p_star_vap_a = math.e**(A - B/(C+T))
    print(f'the pure component vapour pressure at saturation: {p_star_vap_a}') 
    return p_star_vap_a/750.06

def composition_analysis(p_sys,p_star_vap_a,p_star_vap_b):
    x_a = (p_sys - p_star_vap_a)/(p_star_vap_b-p_star_vap_a)
    print(f'the liquid composition is {x_a}')
    return x_a

def relative_volativity_raoults(p_star_vap_a,p_star_vap_b):
    alpha = p_star_vap_a/p_star_vap_b
    return alpha

def distributed_a_in_raffinate(b,S,m,N,x_f):

    X_n = x_f*(b/(b+S*m))**N

    print(f'distributed a is {X_n}')

def Solvent_flowrate(b,x_f,X_n,y_n):
    S = (b*x_f-X_n*b)/y_n
    print(S)

a, b, c = 15.9008, 2788.51, -52.36
a1, b1, c1 = 16.0137, 3096.52, -53.67
Temperature(750.06,a1,b1,c1)