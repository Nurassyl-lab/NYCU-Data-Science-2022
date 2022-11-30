import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fractions import Fraction
from scipy.stats import norm

def update(table):
  """Compute the posterior probabilities."""
  table['unnorm'] = table['prior'] * table['likelihood'] # call the result `unnorm` because these values are the "unnormalized posteriors"
  prob_data = table['unnorm'].sum()
  table['posterior'] = table['unnorm'] / prob_data # Caclulate the normalized posterior probability
  return prob_data


table3 = pd.DataFrame(index=['Door 1', 'Door 2', 'Door 3'])
table3['prior'] = Fraction(1, 3)
table3

table3['likelihood'] = Fraction(1, 2), 1, 0
table3

update(table3)
table3

from numpy import random
mu, sigma = 50, [10,50,100] # mean and standard deviation
plt.figure(figsize = (8,8))
plt.title('PDFs')
x1 = np.random.normal(mu, sigma[0], size=(1000))
x2 = np.random.normal(mu, sigma[1], size=(1000))
x3 = np.random.normal(mu, sigma[2], size=(1000))
plt.hist(x1, bins=100, density=True, histtype = 'step', label = 'sigma 10')
plt.hist(x2, bins=100, density=True, histtype = 'step', label = 'sigma 50')
plt.hist(x3, bins=100, density=True, histtype = 'step', label = 'sigma 100')
plt.legend()

np.random.seed(seed=50)
x = np.empty((1000))

def bernoulli_trials(n, p):
  success = 0
  for i in range(n):
    random_number = np.random.uniform(0.0, 1.0, 1)[0]
  # If less than p, it's a success so add one to n_success
  if random_number < p:
    success += 1
  return success


for i in range(1000):
  x[i] = bernoulli_trials(100,0.05)


plt.figure()
# Plot the histogram with default number of bins; label your axes
plt.hist(x)
plt.xlabel('number of defaults out of 100 loans')
plt.ylabel('probability')
# Show the plot
plt.show()


A = np.array([[4, 8, 1], [1, 7, -3], [2, -3, 2]])
b = np.array([2, -14, 3])
x = np.dot(np.linalg.inv(A), b)
x



A = np.array([[1.3, 0.6, 0.000000000000001], [4.7, 1.5, 0.00000000000001], [3.1 , 5.2, 0.0000000000001]])
b = np.array([3.3, 13.5, -0.1])
x = np.dot(np.linalg.inv(A), b)
print(x[0], x[1])