# Ref. https://stackoverflow.com/questions/20011122/fitting-a-normal-distribution-to-1d-data


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


N = 999999


X = norm.rvs(0, 3, size = N)
Z = norm.rvs(1, 4, size = N)
Y = norm.rvs(1, 5, size = N)

X_Z = X + Z

plt.subplot(1,2,1)
mu, std = norm.fit(Y)
plt.hist(Y, bins=25, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.ylabel('p(y)')
plt.xlabel('y')


plt.subplot(1,2,2)
mu, std = norm.fit(X_Z)
plt.hist(X_Z, bins=25, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.ylabel('p(x + z)')
plt.xlabel('x + z')


plt.show()
