import numpy as np
from scipy.stats import norm
from statsmodels.stats.weightstats import ztest

np.random.seed(0)

N = 100
mu = 0.2
sigma = 1
x = np.random.randn(N) * sigma + mu

#two sided test
print(ztest(x))

# two-sided test
mu_hat = x.mean()
sigma_hat = x.std(ddof=1)
z = mu_hat / (sigma_hat / np.sqrt(N)) # our mu0 = 0
p_right = 1 - norm.cdf(np.abs(z))
p_left = norm.cdf(-np.abs(z))
p = p_right + p_left
print(z, p)

# one-sided test
ztest(x, alternative='larger')

# one-sided test
mu_hat = x.mean()
sigma_hat = x.std(ddof = 1)
z = mu_hat / (sigma_hat / np.sqrt(N)) # our mu0 = 0
p = 1 - norm.cdf(z)
print(z, p)

# null under a different reference value
mu0 = 0.2
ztest(x, value= mu0)

# null under a different reference value
mu_hat = x.mean()
sigma_hat = x.std(ddof = 1)
z = (mu_hat - mu0) / (sigma_hat / np.sqrt(N)) # our mu0 = 0
p_right = 1 - norm.cdf(np.abs(z))
p_left = norm.cdf(-np.abs(z))
p = p_right + p_left
print(z, p)

# null under a different reference
mu_hat = x.mean()
sigma_hat = x.std(ddof=1)
z = (mu_hat - mu0) / (sigma_hat / np.sqrt(N))
p_right = 1 - norm.cdf(np.abs(z))
p_left = norm.cdf(-np.abs(z))
p = p_right + p_left
print("null under a diff reference", z, p)

# two-sample test
N0 = 100
mu0 = 0.2
sigma0 = 1
x0 = np.random.randn(N)*sigma0 + mu0

N1 = 100
mu1 = 0.5
sigma1 = 1
x1 = np.random.randn(N)*sigma1 + mu1

print(ztest(x0, x1))

# two-sample test implementation
mu_hat0 = x0.mean()
mu_hat1 = x1.mean()
dmu_hat = mu_hat1 - mu_hat0
s2_hat0 = x0.var(ddof=1)
s2_hat1 = x1.var(ddof=1)
s_hat = np.sqrt(s2_hat0 / N0 + s2_hat1 / N1)
z = dmu_hat / s_hat # reference value is 0
p_right = 1 - norm.cdf(np.abs(z))
p_left = norm.cdf(-np.abs(z))
p = p_right + p_left
print(z, p)

# show that we will reject the null hypothesis when the
# null hypothesis is true (false alarm) 5% of the time
num_tests = 10000
results = np.zeros(num_tests)
for i in range(num_tests):
    x1 = np.random.randn(100)
    x2 = np.random.randn(100)
    z, p = ztest(x1, x2)
    results[i] = (p < 0.05)
print(results.mean())




