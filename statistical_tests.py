import numpy as np
import math 
import pandas as pd
import scipy
import scipy.stats as st
import matplotlib.pyplot as plt
import statsmodels as sm
from web_scraping import WebCrawler

# t tests of 2 samples will be the same as a 2 sample z test with sigma = smaple mean
def z_test(x_bar=0,mu_0=0,sigma=0,n=0, n1=0, n2=0, sigma1=0, sigma2=0, two_samples=0, alpha=0, type_of_test='null'):
    zscore_alpha = st.norm.ppf(alpha)
    zscore_0 = zscore(x_bar,mu_0,sigma,n, n1, n2,sigma1, sigma2, two_samples)
    pvalue = st.norm.cdf(zscore_0)
    if two_samples == True:
        n = 1
        sigma = (math.sqrt(((sigma1**2)/n1)+(sigma2**2)/n2))
        buttom_bound = x_bar - zscore_alpha*sigma
        higher_bound = x_bar + zscore_alpha*sigma
        confidence_interval = f'the confidence interval is ({higher_bound} < mu < {buttom_bound})'
    else:
        buttom_bound = x_bar - zscore_alpha*sigma/math.sqrt(n)
        higher_bound = x_bar + zscore_alpha*sigma/math.sqrt(n)
        confidence_interval = f'the confidence interval is ({higher_bound} < mu < {buttom_bound})'
    if type_of_test == '2tailed':
        alpha = alpha/2
        zscore_alpha = st.norm.ppf(alpha)
        pvalue = st.norm.cdf(zscore_0)
        pvalue = 2*(1 - pvalue)
        buttom_bound = x_bar - zscore_alpha*sigma/math.sqrt(n)
        higher_bound = x_bar + zscore_alpha*sigma/math.sqrt(n)
        confidence_interval = f'the confidence interval is ({higher_bound} < mu < {buttom_bound})'
        if abs(zscore_alpha) > abs(zscore_0):
            print('we fail to reject the H0')
        else:
            print('we reject the H0')
    elif type_of_test == 'greater':
        zscore_alpha = st.norm.ppf(alpha)
        pvalue = st.norm.cdf(zscore_0)
        pvalue = 1 - pvalue
        if abs(zscore_alpha) > zscore_0:
            print('we fail to reject the H0')
        else:
            print('we reject the H0')
    else:
        zscore_alpha = st.norm.ppf(alpha)
        pvalue = st.norm.cdf(zscore_0)
        if zscore_0 > zscore_alpha:
            print('we fail to reject the H0')
        else:
            print('we reject the H0')
    return zscore_alpha, zscore_0 ,pvalue, confidence_interval

#if you use 1 sample test insert n1 to be your n - 1 or your dof
def t_test(var1=0, var2=0, sample_mean=0, std=0, n1=0, n2=0, popmean=0, samples=0, alpha=0.05, type_of_test='2tailed'):
    n = n1
    tscore_alpha = st.t.ppf(alpha, n - 1)
    if samples == 2:
        std = math.sqrt(((n1 - 1)*var1 + (n2 - 1)*var2)/(n1 + n2 - 2))
        n = 1/n1 + 1/n2
        tscore_alpha2 = st.t.ppf(alpha/2, n1 + n2 - 2)
        tscore_0 = (sample_mean)/(std*math.sqrt(n))
        buttom_bound = sample_mean - tscore_alpha2*std*math.sqrt(n)
        higher_bound = sample_mean + tscore_alpha2*std*math.sqrt(n)
        n = n1 + n2 - 2
        tscore_alpha = st.t.ppf(alpha, n)
        pvalue = st.t.cdf(tscore_0, n)
        confidence_interval = f'the confidence interval is ({higher_bound} < mu < {buttom_bound})'
    else:
        tscore_0 = tscore(sample_mean, popmean, std, n)
        tscore_alpha2 = st.t.ppf(alpha/2, n)
        buttom_bound = sample_mean - tscore_alpha2*std/math.sqrt(n + 1)
        higher_bound = sample_mean + tscore_alpha2*std/math.sqrt(n + 1)
        confidence_interval = f'the confidence interval is ({higher_bound} < mu < {buttom_bound})'
    if type_of_test == '2tailed':
        alpha = alpha/2
        tscore_alpha = st.t.ppf(alpha, n - 1)
        pvalue = st.t.cdf(tscore_0, n)
        pvalue = 2*(1 - pvalue)
        if abs(tscore_alpha) > abs(tscore_0):
            print('we fail to reject the H0')
        else:
            print('we reject the H0')
    elif type_of_test == 'greater':
        pvalue = st.t.cdf(tscore_0, n)
        pvalue = 1 - pvalue
        if abs(tscore_alpha) > tscore_0:
            print('we fail to reject the H0')
        else:
            print('we reject the H0')
    else:
        pvalue = st.t.cdf(tscore_0, n)
        if tscore_0 > tscore_alpha:
            print('we fail to reject the H0')
        else:
            print('we reject the H0')
    return tscore_alpha, tscore_0 ,tscore_alpha2, pvalue, confidence_interval

def chisquared_test(sample_std=0,sigma=0,n=0, alpha=0, type_of_test='null'):
    n = n - 1
    chi2_alpha = st.chi2.ppf(q=alpha,df=n)
    chi2_0 = chisquared(sample_std, sigma,n)
    pvalue = st.chi2.cdf(chi2_0)
    alpha1 = alpha/2
    alpha2 = 1 - alpha1
    alpha3 = 1 - alpha
    chi2_alpha1 = st.chi2.ppf(alpha1)
    chi2_alpha2 = st.chi2.ppf(alpha2)
    chi2_alpha3 = st.chi2.pdf(alpha3)
    buttom_bound = (n-1)*sample_std**2/chi2_alpha1
    higher_bound = (n-1)*sample_std**2/chi2_alpha2
    confidence_interval = f'the confidence interval is ({higher_bound} < mu < {buttom_bound})'
    if type_of_test == '2tailed':
        chi2_0 = chisquared(sample_std, sigma, n)
        pvalue = st.chi2.cdf(chi2_0)
        pvalue = 2*(1 - pvalue)
        chi2_alpha = chi2_alpha1
        if chi2_0 > chi2_alpha1 or chi2_0 < chi2_alpha2:
            print('we reject the H0')
        else:
            print('we fail to reject the H0')
    elif type_of_test == 'greater':
        pvalue = st.chi2.cdf(chi2_0)
        pvalue = 1 - pvalue
        if chi2_alpha > chi2_0:
            print('we fail to reject the H0')
        else:
            print('we reject the H0')
    else:
        chi2_alpha = chi2_alpha3
        pvalue = st.norm.cdf(chi2_0)
        if chi2_0 > chi2_alpha3:
            print('we fail to reject the H0')
        else:
            print('we reject the H0')
    return chi2_alpha, chi2_alpha2, chi2_0 ,pvalue

def f_test(var1, var2, n1, n2, alpha=0.05, type_of_test='2tailed'):
    n1 = n1 - 1
    n2 = n2 - 1
    f = ff(var1, var2, alpha, n1, n2)
    f_0 = f[0]
    f_alpha1 = f[1]
    f_alpha2 = f[2]
    f_alpha3 = f[3]
    buttom_bound = f_0*f_alpha3
    higher_bound = f_0*f_alpha1
    confidence_interval = f'the confidence interval is ({higher_bound} < mu < {buttom_bound})'
    print(confidence_interval)
    if type_of_test == '2tailed':
        if f_0 > f_alpha1 or f_0 < f_alpha2:
            print('we reject the H0')
        else:
            print('we fail to reject the H0')
    elif type_of_test == 'greater':
        if f_0 > f_alpha1:
            print('we reject the H0')
        else:
            print('we fail to reject the H0')
    else:
        if f_0 < f_alpha2:
            print('we reject the H0')
        else:
            print('we fail to reject the H0')
    return f_0, f_alpha1, f_alpha2, f_alpha3

#implement a webcrawler that scans over the internet to look for news to trade 
# greater fools theory states that the price of a stock does not move according to its intrinsic value but rather
# due to its participants being scared or confident in the assests future 
# thi8s class is based on the theory that there is no fundamental analyhsis that is valid 
# and you only need to look at market news to trade the stock

# make Adam the signal selection bot be able to not only recognise when returns come from one strategy rather then the other
# but also implement adavanced portfolio optimisation techniques  

class GreaterFoolTheory(WebCrawler):
    def __init__(self, website):
        self.website = website
    