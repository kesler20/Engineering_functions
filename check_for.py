import numpy as np
import math 
from matplotlib import pyplot as plt 

Ts = 100
T0 = 25
x = [i for i in range(0,101)]
alpha = 8.1*10**(-8)
t = 1
T = [Ts + (T0 - Ts)*math.erf(x_i/(2*np.sqrt(alpha*t))) for x_i in x]
print(T)