import sympy
from sympy import Symbol, Derivative
import math

'''
 in order to use function Stat_point_evaluation you need to instantiate the class Symbol()
 and you can pass the polynomial directly onto the arguments of the function 
 bare in mind that x^2 is replaced with x**2 

 sample code :
    x= Symbol('x')
    y= Symbol('y') 
    second_derivative_test(x*y + 64/y + 64/x, 4, 4)
 
 the bisection function takes in the value of a string representing the function 
 sample code :
    
    func = 'x**3+3*x**2+5*x+9'
    bisection_method(func, -2, -3, 0.01)

 where 2 is the upper bound and 1 is the lower bound and the interval is 0.001
 this will also return the number of steps computed
 
 The Newton raphson method function or N-R takes in a function as a string, 
 initial guess and the boundaries of the inial guess as well as the accuracy
 
 sample code:
    function = '2.718281828459045**(2*x) - 25*x + 10'
    Newton_Raphson_method(function, 1.5, 3, 1, 0.0001)
 
 the small error function can be used for both 3 variables and bivariate functions 
 just set the z and dz to 0 when you want to do on f(x,y)
 
 sample code:
 function = '(x**2 + y**2)**0.5'
 small_error_aproxx(function,0.003, 0.003, 0, 1, 2, 0)
 '''


def second_derivative_test(function, stat_point_x, stat_point_y): 
    x= Symbol('x')
    y= Symbol('y')
    partialderiv= Derivative(function, x)
    f_x = partialderiv.doit()
    f_y = Derivative(function, y).doit()

    f_xx = Derivative(f_x, x).doit()
    f_xy = Derivative(f_x, y).doit()
    f_yy = Derivative(f_y, y).doit()
    print(f'''the derivatives are: 
    f_x = {f_x} 
    f_y = {f_y} 
    f_xx = {f_xx} 
    f_xy = {f_xy} 
    f_yy = {f_yy}''')
    
    x, y = stat_point_x, stat_point_y
    xx = eval(str(f_xx))
    yy = eval(str(f_yy))
    xy = eval(str(f_xy))
    
    print(f'''the derivatives are:  
    f_xx = {xx} 
    f_xy = {xy} 
    f_yy = {yy}''')

    delta = float(xx)*float(yy) - float(xy^2)
    if delta < 0:
        print('you have a saddle point')
    elif delta > 0:
        if xx > 0:
            print('you have a minimum point')
        else:
            print('you have a maximum point')
    else:
        print('test was inconclusive')

def bisection_method(func, upper_bound, lower_bound, accuracy):
    
    midpoint = (upper_bound + lower_bound)/2
    x = upper_bound
    f_a = eval(func)
    x = midpoint
    f_b = eval(func)
    flag = f_b*f_a
    a = lower_bound
    b = upper_bound
    n_ = (math.log((b-a)/accuracy, math.e))/(math.log(2, math.e))
    n_iterations = round(n_)
    
    for i in range(0,n_iterations + 1):
        if flag < 0:
            lower_bound = midpoint
            m1 = (upper_bound + lower_bound)/2
            x = upper_bound
            f_a = eval(func)
            x = m1
            f_b = eval(func)
            flag = f_b*f_a
            midpoint = m1
        else:
            upper_bound = midpoint
            m1 = (upper_bound + lower_bound)/2
            x = upper_bound
            f_a = eval(func)
            x = m1
            f_b = eval(func)
            flag = f_b*f_a
            midpoint = m1    
    print(f'{(upper_bound, lower_bound)}, {n_iterations}')

def Newton_Raphson_method(f, initial_guess, b, a, accuracy):
    x= Symbol('x')
    y= Symbol('y')
    f_x = Derivative(f, x).doit()
    n_ = (math.log((b-a)/accuracy, math.e))/(math.log(2, math.e))
    n_iterations = round(n_)
    print(f_x)
    for i in range(0, n_iterations + 1):  
        f_x = str(f_x)
        x = initial_guess
        xx = eval(f_x)
        function = eval(f)
        ans = x - (float(function))/(float(xx))
        initial_guess = ans
    print(ans)


def small_error_aproxx(function,dx,dy,dz,x1,y1,z1):
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    
    df_x = Derivative(function, x).doit()
    df_y = Derivative(function, y).doit()
    df_z = Derivative(function, z).doit()
    
    df_x = str(df_x)
    df_y = str(df_y)
    df_z = str(df_z)
    
    x = x1
    y = y1
    z = z1

    print(f'''
    df_x = {df_x}
    df_y = {df_y}
    df_z = {df_z}''')
    
    df_x = eval(df_x)
    df_y = eval(df_y)
    df_z = eval(df_z)
    
    df = df_x*dy + df_y*dy + df_z*dz
    func = eval(function)
    print(f' the maxiimum resulting error is {df}')
    prcnt_error = df/func
    error = prcnt_error*100
    print(f'% error = {error}')


def linearization_technique(function,steady_state_value):
    x = Symbol('x')
    y = Symbol('y')
    f_x = Derivative(function, x).doit()

    print(f'the derivative of the function is {f_x}')
    f_x = str(f_x)
    x = steady_state_value
    f_xs = eval(function)
    print(f'the function evaluated at steady state is {f_xs}')
    f_x = eval(f_x)
    print(f'the derivative of the function evaluated at steady state is {f_x}')
    deviation_variable = y - x
    
    print(f'the deviation variable is: {deviation_variable}')
    f = f_xs + f_x*deviation_variable

    print(f' the linearised function is {f}')


function = '(300*2.718281828459045**(-0.1*x))/(x**2)'
linearization_technique(function, 10)