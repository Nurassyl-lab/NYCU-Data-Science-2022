import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
import seaborn as sns
sns.set()
np.random.seed(523)


# In[1]
def f1(x):
  a = 3.2
  b = 5.6
  beta = gamma(a) * gamma(b) /gamma(a + b)
  p = x ** (a - 1) * (1-x) ** (b - 1)
  return 1/beta * p


mode = (3.2-1)/(3.2+5.6-2)
c = f1(mode)


def beta_gen(n):
  i = 0
  output = np.zeros(n)
  while i < n:
    U = np.random.uniform(size = 1)
    V = np.random.uniform(size = 1)
    if U < 1/c * f1(V):
      output[i] = V
      i = i + 1
  return output


px = np.arange(0,1+0.01,0.01)
py = f1(px)


Y = beta_gen(n = 1000)
fig,ax = plt.subplots()
temp = ax.hist(Y,density=True)
ax.plot(px,py)
plt.title("Beta(3.2, 5.6) || Example 1")
plt.show()

# In[2]
def beta_gen2(n): 
  i = 0
  output = np.zeros(n)
  while i < n:
    U = np.random.uniform(size = 2 + 6)
    p1 = np.sum(np.log(U[0:2]))
    p2 = np.sum(np.log(U))
    output[i] = p1/p2
    i = i + 1
  return output


##PDF of Beta(2,6)
def f2(x):
  a = 2
  b = 6
  beta = gamma(a) * gamma(b) /gamma(a + b)
  p = x ** (a - 1) * (1-x) ** (b - 1)
  return 1/beta * p


def beta_gen3(n):
    i = 0
    mode_Y = (3.2 - 1)/ (3.2+5.6-2)
    # mode_V = (2 - 1)/ (2+6-2)
  
    # M = f1(mode_Y) / f2(mode_Y)
    M = np.max([f1(mode_Y), f2(mode_Y)])
    print(M)
    print(f1(mode_Y) / (f2(mode_Y) * M))
    output = np.zeros(n)
    while i < n:
        U = np.random.uniform(size = 1)
        V = beta_gen2(1)
        if U < (1/M) * (f1(V)/f2(V)):
            output[i] = V
            i = i + 1
    return output


Y = beta_gen3(n = 1000)
fig,ax = plt.subplots()
temp = ax.hist(Y,density=True)
ax.plot(px,py)
plt.title("Beta(3.2, 5.6) || Example 2")
plt.show()

# In[3]
gss = pd.read_csv('gss.csv', index_col=0)
gss.head()


banker = gss.loc[:, 'indus10'] == 6870
banker.head()


non_bankers, bankers = banker.value_counts()
print(f'In this dataset, there are {bankers} bankers')


banker_frac = banker.value_counts(normalize = True)[1]
print(f'About {banker_frac * 100}% of the respondents work in banking, so if we choose a random person from the dataset, the probability they are a banker is about {banker_frac * 100}%.')


def prob(A):
    """Computes the probability of a proposition, A."""
    return A.value_counts(normalize = True)[1] * 100
prob(banker)


female = gss.loc[:, 'sex'] == 2
prob(female)


liberal = gss.loc[:, 'polviews'] <= 3 
print(f'If we choose a random person in this dataset, the probability they are liberal is about {prob(liberal)}%.')


democrat = gss.loc[:, 'partyid'] <= 1
prob(democrat)


prob(banker)
prob(democrat)


prob(democrat * banker)


selected = gss.loc[gss['polviews'] <= 3]
print(str(prob(selected.loc[:, 'partyid'] <= 1)) + '%')


selected = gss.loc[gss['indus10'] == 6870]
tmp = prob(selected.loc[:, 'sex'] == 2)
print(f'About {tmp}% of the bankers in this dataset are female.')


def conditional(proposition, given):
    return prob(proposition*given)

a = gss.loc[gss['sex'] == 2, 'sex'] == 2
b = gss.loc[gss['sex'] == 2, 'polviews'] <= 3
print(f'About {conditional(b, a)}% of female respondents are liberal.')


a = gss.loc[gss['indus10'] == 6870, 'indus10'] == 6870
b = gss.loc[gss['indus10'] == 6870, 'sex'] == 2
print(f'{conditional(b, a)}%')


a = gss.loc[gss['sex'] == 2, 'sex'] == 2
b = gss.loc[gss['sex'] == 2, 'indus10'] == 6870
print(f'Only about {conditional(b, a)}% of female respondents are bankers.')


a = gss.loc[gss['polviews'] <= 3]
b = a.loc[a['partyid'] <= 1] 
c = b.loc[:, 'partyid'] <= 1
d = b.loc[:, 'sex'] == 2
print(f'About {conditional(d, c)}% of liberal Democrats are female.')


a = gss.loc[gss['indus10'] == 6870]
b = a.loc[a['partyid'] <= 1]
c = a.loc[:, 'partyid'] <= 1
d = b.loc[:, 'sex'] == 2
print(f'About {conditional(d,c)}% of bankers are liberal women.')