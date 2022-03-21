import time
import starting_projects
from starting_projects.common_func import Convert
C_max = 31
a = 0.75
total_flowrate_CH_out_of_FT = 300
output_product_number_of_C = 4

def mass_balance_on_distillation(C_max, a, total_flowrate_CH_out_of_FT, output_product_number_of_C):
    #the Convert function converts a list of touples into a dictionary 
    products = {}
    product_distribution = []
    
    for n in range(C_max + 1):
        if n == 0:
            pass
        else:
            W_n = n*(((1 - a)**2)*(a**(n-1)))
            weights = (n, W_n)
            product_distribution.append(weights)
    output_distribution = Convert(product_distribution, products)
    X = []
    for n in range(1,C_max):
        x = total_flowrate_CH_out_of_FT/n
        x = x*output_distribution[n][0]
        X.append(x)
    total_flowrate_CH_adjusted = sum(X)
    print(f" flowrate of hydrocarbons leaving FT: {total_flowrate_CH_adjusted} kmoles/hr")

    flowrates = []
    for i in range(1,C_max + 1):
        h_n = 2*i + 2
        x_a = output_distribution[i][0]
        C_n = x_a*total_flowrate_CH_adjusted
        print(f"C({i})H({h_n}) : {C_n} kmoles/hr")
        flowrates.append(C_n)

    kjl = []
    for i in range(len(flowrates)):
        flow = flowrates[i]
        if i <= (output_product_number_of_C - 2):
            kjl.append(flow)
        else:
            pass
    print(f"the product flowrate is : {sum(kjl)} kmoles/hr")
    return sum(kjl), total_flowrate_CH_adjusted

x , y = mass_balance_on_distillation(C_max, a, total_flowrate_CH_out_of_FT, output_product_number_of_C)
