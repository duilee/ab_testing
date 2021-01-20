import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(0)

mu = 170
sd = 7

# generate samples from our distribution
x = norm.rvs(loc=mu, scale=sd, size=100)

x.mean() # maximum likelihood mean
x.var() # maximum likelihood variance

x.std() # maximum likelihood std

x.std(ddof=1) # unbiased std

# 95 percentile
norm.ppf(0.95, loc=mu, scale=sd)

# if you are 160 cm tall, what percentile are you
norm.cdf(160, loc=mu, scale=sd)



