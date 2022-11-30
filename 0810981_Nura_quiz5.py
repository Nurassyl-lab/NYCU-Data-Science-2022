import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

'First'
'a'
prob = 1-norm.cdf(600,500,75)# assuming that cdf is sum of PDFs from START to 599
print(prob)

'b'
mu = 500
sigma = 75
x = np.arange(200,800)
prob = norm.pdf(x,mu,sigma)
plt.figure()
plt.plot(x,prob);
plt.title(r'$\mathrm{N(\mu=500, \sigma^2=75^2)}$')
#plt.ylim((0,0.006))
plt.ylim((0,0.006))
plt.show()

'c'
x = [3,5,10]
sigma = 2
mu = np.arange(4,8,0.01)
like = norm.pdf(x[0],mu,sigma)*norm.pdf(x[1],mu,sigma)*norm.pdf(x[2],mu,sigma)
plt.plot(mu,like,color="darkred");
plt.title('Likelihood Function')
plt.xlabel(r'$\mu$')
plt.show()

'd'
mle = mu[np.argmax(like)]
print(mle)

'e'
x = np.arange(200,800)
prob_hat = sum(prob[:400])
print(1 - prob_hat)

'f'
CDF = norm.cdf(x,500,75)
plt.figure()
plt.title('(f) CDF of norm variable x')
plt.plot(x, CDF)




'Second'
df = pd.read_csv('Advertising_adj.csv')
def bootstrap(df):
    selectionIndex = np.random.randint(20, size = 20)
    new_df = df.iloc[selectionIndex]
    return new_df

beta0_list, beta1_list = [],[]

number_of_bootstraps = 1000
for i in range(number_of_bootstraps):
    df_new = bootstrap(df)

    x = pd.Series(df_new['tv'])
    y = pd.Series(df_new['sales'])

    xmean = x.mean()

    ymean = y.mean()

    beta1 = np.dot((x-xmean) , (y-ymean))/((x-xmean)**2).sum()
    beta0 = ymean - beta1 * xmean

    beta0_list.append(beta0)
    beta1_list.append(beta1)

beta0_mean = np.mean(beta0_list)
beta1_mean = np.mean(beta1_list)

fig, ax = plt.subplots(1,2, figsize=(18,8))
ax[0].hist(beta0_list)
ax[1].hist(beta1_list)
ax[0].set_xlabel('Beta0')
ax[1].set_xlabel('Beta1')
ax[0].set_ylabel('Frequency');




'Third'
beta0_list, beta1_list = [],[]
numberOfBootstraps = 100
for i in range(numberOfBootstraps):
    df_new = bootstrap(df)
    xmean = df_new.tv.mean()
    ymean = df_new.sales.mean()
    beta1 = np.dot((df_new.tv-xmean) , (df_new.sales-ymean))/((df_new.tv-xmean)**2).sum()
    beta0 = ymean - beta1*xmean
    beta0_list.append(beta0)
    beta1_list.append(beta1)

beta0_list.sort()
beta1_list.sort()

beta0_CI = (np.percentile(beta0_list, 2.5),np.percentile(beta0_list, 97.5))
beta1_CI = (np.percentile(beta1_list, 2.5),np.percentile(beta1_list, 97.5))

print(f'The beta0 confidence interval is {beta0_CI}')
print(f'The beta1 confidence interval is {beta1_CI}')

def plot_simulation(simulation,confidence):
    plt.figure()
    plt.hist(simulation, bins = 30, label = 'beta distribution', align ='left', density = True)
    plt.axvline(confidence[1], 0, 1, color = 'r', label = 'Right Interval')
    plt.axvline(confidence[0], 0, 1, color = 'red', label = 'Left Interval')
    plt.xlabel('Beta value')
    plt.ylabel('Frequency')
    plt.title('Confidence Interval')
    plt.legend(frameon = False, loc = 'upper right')
    
plot_simulation(beta0_list, beta0_CI)
plot_simulation(beta1_list, beta1_CI)

