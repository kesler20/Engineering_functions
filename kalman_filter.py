from pykalman import KalmanFilter, UnscentedKalmanFilter
import numpy as np
from numpy import ma
from universe_simplifiations import get_daily_returns, series_to_list
from matplotlib import pyplot as plt
import pandas as pd
#github repository at : https://pykalman.github.io/

#Like KalmanFilter, two methods are provided 
#in UnscentedKalmanFilter for tracking targets: 
# UnscentedKalmanFilter.filter() and UnscentedKalmanFilter.smooth(). 

kf = KalmanFilter(
    transition_matrices=None, 
    observation_matrices=None, 
    transition_covariance=None, 
    observation_covariance=None, 
    transition_offsets=None, 
    observation_offsets=None, 
    initial_state_mean=None, 
    initial_state_covariance=None, 
    random_state=None, 
    em_vars=['transition_covariance', 'observation_covariance', 'initial_state_mean', 'initial_state_covariance'], 
    n_dim_state=None, 
    n_dim_obs=None
)

kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

measurements = ma.asarray(measurements)
measurements[1] = ma.masked   # measurement at timestep 1 is unobserved
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, observation_covariance=0.1)
(filtered_state_means, filtered_state_covariances) = ukf.filter([0, 1, 2])
(smoothed_state_means, smoothed_state_covariances) = ukf.smooth([0, 1, 2])

def f(state, noise):
    return state + np.sin(noise)
def g(state, noise):
    return state + np.cos(noise)

ukf = UnscentedKalmanFilter(f, g)
ukf.smooth([0, 1, 2])[0]

ticker = 'TSLA'
price_data = get_daily_returns(ticker, 5)
price_data = pd.DataFrame(price_data)

price_data1 = series_to_list(price_data['close'][:round(len(price_data)/2)])
price_data2 = series_to_list(price_data['close'][round(len(price_data)/2):len(price_data)])
price_data0 = []
for i in range(len(price_data1)):
    price_data0.append([price_data1[i], i])
price_data = []
for i in range(len(price_data2)):
    price_data.append([price_data2[i], i])

measurements = price_data 
filtered = []
for i in range(len(measurements)):
    kf = KalmanFilter(observation_matrices=price_data0[i])
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements[i])
    #(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurement)
    filtered_state_means = series_to_list(filtered_state_covariances[0])
    filtered.append(filtered_state_means)

filtered0 = []
for i in filtered:
    filtered0.append(i[0])
filtered0 = pd.DataFrame(filtered0)
filtered0.plot()
plt.show()