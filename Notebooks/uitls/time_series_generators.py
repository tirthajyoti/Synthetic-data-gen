import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

def generate_bell(length, amplitude, default_variance):
    bell = np.random.normal(0, default_variance, length) + amplitude * np.arange(length)/length
    return bell

def generate_funnel(length, amplitude, default_variance):
    funnel = np.random.normal(0, default_variance, length) + amplitude * np.arange(length)[::-1]/length
    return funnel

def generate_cylinder(length, amplitude, default_variance):
    cylinder = np.random.normal(0, default_variance, length) + amplitude
    return cylinder

std_generators = [generate_bell, generate_funnel, generate_cylinder]

def generate_pattern_data(length=100, avg_pattern_length=5, avg_amplitude=1, 
                          default_variance = 1, variance_pattern_length = 10, variance_amplitude = 2, 
                          generators = std_generators, include_negatives = True):
    """
    Generates 1-D time sereies using compositions of randomized series and customized patterns
    
    Example
    ------------
    n_data= [50, 150, 500]
    n_pattern_length = [5,10,20]
    from itertools import product
    config_ = list(product(n_data,n_pattern_length))
    fig, axes = plt.subplots(nrows=3,ncols=3,figsize=(12,9))
    ax = axes.ravel()
    i=0
    for n1,n2 in config_:
        data = generate_pattern_data(length=n1,avg_pattern_length=n2)
        ax[i].plot(data,color='k')
        ax[i].grid(True)
        i+=1
    plt.show()
    """
    data = np.random.normal(0, default_variance, length)
    current_start = random.randint(0, avg_pattern_length)
    current_length = current_length = max(1, math.ceil(random.gauss(avg_pattern_length, variance_pattern_length)))
    
    while current_start + current_length < length:
        generator = random.choice(generators)
        current_amplitude = random.gauss(avg_amplitude, variance_amplitude)
        
        while current_length <= 0:
            current_length = -(current_length-1)
        pattern = generator(current_length, current_amplitude, default_variance)
        
        if include_negatives and random.random() > 0.5:
            pattern = -1 * pattern
            
        data[current_start : current_start + current_length] = pattern
        
        current_start = current_start + current_length + random.randint(0, avg_pattern_length)
        current_length = max(1, math.ceil(random.gauss(avg_pattern_length, variance_pattern_length)))
    
    return np.array(data)

def gen_series_anomaly(size=1000,
                    anomaly_frac=0.02,anomaly_scale=2.0,
                    loc=0.0,scale=1.0):
    """
    Generates a time-series data (array) with some anomalies
    
    Arguments:
        size: Size of the array
        anomaly_frac: Fraction anomalies
        anomaly_scale: Scale factor of anomalies
        loc: Parameter (mean) for the underlying Gaussian distribution
        scale: Parameter (std.dev) for the underlying Gaussian distribution
    """

    arr = np.random.normal(loc=loc,scale=scale,size=size)
    arr_min = arr.min()
    arr_max = arr.max()
    no_anomalies = int(size*anomaly_frac)
    idx_list=np.random.choice(a=size,size=no_anomalies,replace=False)
    for idx in idx_list:
        arr[idx] = loc+np.random.uniform(low=arr_min-anomaly_scale*(arr_max-arr_min),high=arr_max+anomaly_scale*(arr_max-arr_min))
    return arr

def gen_ts_dataframe(n=10,prob_anomolous=0.1,
                    size=1000,anomaly_frac=0.02,anomaly_scale=2.0,
                    loc=0.0,scale=1.0):
    """
    Generates dataframe of time-series containing 'normal' and 'anomolous' samples
    
    Arguments:
        n: Number of time-series
        prob_anomolous: Probability a time-series containing anomalies
        size: Size of the array (individual time-series)
        anomaly_frac: Fraction anomalies
        anomaly_scale: Scale factor of anomalies
        loc: Parameter (mean) for the underlying Gaussian distribution
        scale: Parameter (std.dev) for the underlying Gaussian distribution
    
    Returns:
        A dataframe of shape (n,2) where the first column contains time-series data as list 
        and the second column contains the binary classification of 0 (normal) or 1 (anomolous) 
    """
    assert prob_anomolous < 1.0, print("Probability of anomaly cannot be equal to or greater than 1.0")
    
    dt = {}
    for i in range(n):
        anomolous = np.random.uniform()
        if anomolous < prob_anomolous:
            dt[str(i)] = [gen_series_anomaly(size=size,
                                     anomaly_frac=anomaly_frac,anomaly_scale=anomaly_scale,
                                     loc=loc,scale=scale),1]
        else:
            dt[str(i)] = [gen_series_anomaly(size=size,
                                     anomaly_frac=0.0,anomaly_scale=anomaly_scale,
                                     loc=loc,scale=scale),0]
    df = pd.DataFrame(dt).T
    df.columns = ['ts','anomolous']
    return df