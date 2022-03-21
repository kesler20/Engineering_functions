import numpy as np
import math 
import pandas as pd
import scipy
import scipy.stats as st
import matplotlib.pyplot as plt
import statsmodels as sm
#we fail to reject when the sample mean is not 
# statistically significantly far away from the population mean
# to say that the mean of the sample is not equal to the population mean
# i.e. when we get x_bar = 67 and mu_0 = 70 kg
#fail to reject is equivalent to accept the H0
def error_propagation(linear_combination, x1=0, x2=0, sigma_x1=0, sigma_x2=0, k=0, c1=0, c2=0, sigma_c1=0): #sigma_c2):
    sigma_x1 = sigma_x1/2
    sigma_x2 = sigma_x2/2
    if linear_combination == 'k1':
        mu_y = x1 + k
        sigma_y2 = math.sqrt(sigma_x1**2)
    elif linear_combination == 'k2':
        mu_y = x1*k
        sigma_y2 = math.sqrt((k*sigma_x1)**2)
    elif linear_combination == 'c':
        mu_y = x1*c1 + x2*c2
        sigma_y2 = math.sqrt((x1*sigma_c1)**2 + (sigma_x1*c1)**2) #+ (x2*sigma_c2)**2 + (sigma_x2*c2)**2 ?
    elif linear_combination == 'x1 + x2':
        mu_y = x1 + x2
        sigma_y2 = math.sqrt(sigma_x1**2 + sigma_x2**2)
    else:
        mu_y = x1*x2
        sigma_y2 = math.sqrt((sigma_x1*x2)**2 + (sigma_x2*x1)**2)
    return mu_y, sigma_y2

def zscore(x_bar,mu_0,sigma,n, n1, n2,sigma1=0, sigma2=0, two_samples=False):
    if two_samples == True:
        zscore0 = (x_bar - mu_0)/(math.sqrt(((sigma1**2)/n1)+(sigma2**2)/n2))
    else:
        zscore0 = (x_bar - mu_0)/(sigma/math.sqrt(n))
    return zscore0

def tscore(x_bar,mu_0,sample_sigma,n):
    tscore0 = (x_bar - mu_0)/(sample_sigma/math.sqrt(n))
    return tscore0

def fscore(var1, var2):
    fscore = var1/var2
    return fscore

def ff(alpha, n1, n2, var1=1, var2=1):
    f_0 = fscore(var1, var2)
    alpha1 = np.array(alpha)
    alpha2 = np.array(alpha/2)
    alpha3 = np.array(1 - alpha2)
    f_alpha1 = st.f.ppf(alpha1, n1, n2)
    f_alpha2 = st.f.ppf(alpha2, n1, n2)
    f_alpha3 = st.f.ppf(alpha3, n1, n2)
    return [f_0, f_alpha1, f_alpha2, f_alpha3]

def chisquared(sample_std,sigma,n):
    chisquared0 = ((n - 1)*sample_std**2)/sigma**2
    return chisquared0

def f_test():
    f_alpha = ff(0.975, 4, 4)
    f_alpha = 1/f_alpha[1]
    print(f_alpha)

def ztest():
    zscore_0 = zscore()
    zscore_alpha = st.norm.ppf(0.75)
    print(zscore_alpha)
    pvalue = st.norm.cdf(1.333)
    print(pvalue)

    zscore_alpha = st.norm.ppf(alpha)
    pvalue = st.norm.cdf(zscore_0)
    pvalue = 2*(1 - pvalue)

    zscore_alpha = st.norm.ppf(0.75)
    print(zscore_alpha)

    pvalue = st.norm.cdf(2)
    print(pvalue)


    zscore_alpha = st.norm.ppf(alpha)
    pvalue = st.norm.cdf(zscore_0)

def ttest():
    n = 10
    n1 = 10
    n2 = 10
    sample_mean = 10
    popmean = 1
    std = 1

    tscore_0 = tscore(sample_mean, popmean, std, n)
    tscore_alpha2 = st.t.ppf(alpha/2, n1 + n2 - 2)

    tscore_alpha = st.t.ppf(0.025, 13)
    print(tscore_alpha)
    pvalue = st.t.cdf(tscore_0, n)

    tscore_alpha2 = st.t.ppf(alpha/2, n)

    tscore_alpha = st.t.ppf(alpha, n - 1)
    pvalue = st.t.cdf(tscore_0, n)
    pvalue = 2*(1 - pvalue)

    pvalue = st.t.cdf(tscore_0, n)
    pvalue = 1 - pvalue

    pvalue = st.t.cdf(tscore_0, n)
#sample pooled standard deviation 
# only use if we cannot assume that sigma1 = sigma2 = sigma
#else use the z statistic where sigma is constant

def two_sample():
    n_1 = 8
    n_2 = 8
    S_1 = 2.39
    S_2 = 2.98
    s_p = math.sqrt(((n_1 - 1)*S_1**2 + (n_2 - 1)*S_2**2)/((n_1 - 1)+(n_2 - 1)))
    print(s_p)
    X_bar1 = 92.255
    X_bar2 = 92.733
    t_0 = (X_bar1 - X_bar2)/(s_p*math.sqrt(1/n_1 + 1/n_2))
    print(t_0)
#chi-test
def chitest():
    sample_std = 10 
    sigma = 10
    n = 10
    chi2_alpha = st.chi2.ppf(q=alpha,df=n)
    chi2_0 = chisquared(sample_std, sigma,n)
    pvalue = st.chi2.cdf(chi2_0)

    chi2_alpha1 = st.chi2.ppf(alpha)

    chi2_0 = chisquared(sample_std, sigma, n)

    pvalue = st.chi2.cdf(chi2_0)
    pvalue = 1 - pvalue

    pvalue = st.norm.cdf(chi2_0)

def two_sample_data(x_bar1, x_bar2,mu1, mu2):
    x_bar = x_bar1 - x_bar2
    mu = mu1 - mu2
    return x_bar, mu

def z__0(series):
    z = []
    for measurement in series:
        zscore = (measurement - series.mean())/np.std(series)
        z.append(zscore)
    return z

def roling_zscore(S1, S2):
    rolling_beta = pd.ols(y=S1, x=S2, window_type='rolling', window=30)
    spread = S2 - rolling_beta.beta['x']*S1

    spread_mavg3 = pd.rolling_mean(spread, window=3)
    spread_mavg30 = pd.rolling_mean(spread, window=30)
    return (spread_mavg3 - spread_mavg30)/np.std(spread_mavg30)

def generate_cointegrated_pairs(plot, N):
    beta0 = N/2
    returns = np.random.normal(0,1,N)
    X = pd.Series(np.cumsum(returns)) + beta0
    some_noise = np.random.normal(0, 1 , N)
    Y = X + some_noise + 5
    #adding five so they are not one on top of the other
    Y.name = 'Y'
    if plot == True:
        pd.concat([X, Y], axis=1).plot()
        plt.show()