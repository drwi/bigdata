# coding: utf-8

#%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
import scipy
import sklearn
from sklearn import preprocessing
print ('Generate n = 500 samples from a Beta distribution with parameter a,b=  2, 5.')
a=2
b=5
np.random.seed(2)
#print np.random.get_state()
#raw_input()
cinq_cent_samples=np.random.beta(a, b,size=2)

print(len(cinq_cent_samples))

cinq_cent_samples=np.random.beta(a, b,size=500)
print ("Display the histogram of this sample with 25 bins 1")
if 0:
    fig = plt.figure(figsize=(8, 6))
    plt.hist(cinq_cent_samples, bins=25, normed=True, align='mid')

    #sns.kdeplot(cinq_cent_samples, shade=True, color="b")

    #sns.kdeplot(panda_array_meres, shade=True, color='#9932cc')

    plt.title(' Histogramme/ KDE : 500 samples of beta distrib. ' )
    ax = plt.gca()
    ax.legend_ = None
    plt.xlabel('Valeurs '), plt.ylabel('Frequences')
    plt.tight_layout()
    plt.show()

#for a in range(500):
X=np.random.beta(a, b,size=(2,500))
print (X.shape)
print (np.mean(X,axis=1))

n=500
B=500
#Xstarbarmem = np.zeros((B,1))
cinq_cent_estimators_of_the_mean=[]
for b in range(B):
    #    print np.random.randint(n, size=3)

    Xstar=X[:,np.random.randint(n, size=n)]
    #print Xstar.shape
    #print np.mean(Xstar,axis=1)
    #print 'np.mean(Xstar,axis=1)'
    cinq_cent_estimators_of_the_mean.append(np.mean(Xstar,axis=1))
    #print     X[:,np.random.randint(n, size=4)]
    #raw_input()
    #    Xstarbarmem[:, b] = np.mean(Xstar,axis=1)
    #print X_star
    #print 'raw_input()'
    #raw_input()

#Xstarbarmem = np.sqrt(n) * (Xstarbarmem - np.mean(X,axis=1))
#density_boot = gaussian_kde(Xstarbarmem)
#print Xstarbarmem
#print Xstarbarmem.shape
# print cinq_cent_estimators_of_the_mean[0].shape
# print "cinq_cent_estimators_of_the_mean[0].shape"
# print cinq_cent_estimators_of_the_mean[3].shape
# print "cinq_cent_estimators_of_the_mean[3].shape"
# print cinq_cent_estimators_of_the_mean[0]
# print "cinq_cent_estimators_of_the_mean[0]"
# print X
# print "X"
# raw_input()
if 0:
    fig = plt.figure(figsize=(8, 6))
    for a in range(B):
        if a ==0:
            plt.plot([cinq_cent_estimators_of_the_mean[a][0]],[cinq_cent_estimators_of_the_mean[a][1]],'*', color='b', label='500 estimates of the mean')
            #plt.plot(cinq_cent_estimators_of_the_mean[a][0],cinq_cent_estimators_of_the_mean[a][1],'*', color='b', label='500 estimates of the mean')
            plt.plot([X[0][a]],[X[1][a]],'*', color='r', label='observed data')
        else:
            plt.plot([cinq_cent_estimators_of_the_mean[a][0]],[cinq_cent_estimators_of_the_mean[a][1]],'*', color='b')
            #    print "    plt.plot(cinq_cent_estimators_of_the_mean[a][0],cinq_cent_estimators_of_the_mean[a][1],'*')"
            plt.plot([X[0][a]],[X[1][a]],'*', color='r')
    plt.plot(np.mean(X,axis=1)[0],np.mean(X,axis=1)[1],'*', color='k', label='mean')
    plt.title(' R2 : btstrp estimates of mean' )
    ax = plt.gca()
    ax.legend_ = None
    plt.xlabel('x '), plt.ylabel('y')
    plt.legend(numpoints=1, loc=2)  # numpoints = 1 for nicer display

    plt.tight_layout()
    plt.show()

biases=[  [cinq_cent_estimators_of_the_mean[a][0]-np.mean(X,axis=1)[0],  cinq_cent_estimators_of_the_mean[a][1]-np.mean(X,axis=1)[1]] for a in range(len(cinq_cent_estimators_of_the_mean))]

#print np.mean(cinq_cent_estimators_of_the_mean,axis=1)
q3=0
if q3:
    print "np.mean(cinq_cent_estimators_of_the_mean,axis=1)"
    print np.mean(cinq_cent_estimators_of_the_mean,axis=0).shape
    print np.mean(cinq_cent_estimators_of_the_mean,axis=0)-np.mean(X,axis=1)
    print "np.mean(cinq_cent_estimators_of_the_mean,axis=0)-np.mean(X,axis=1)"
    raw_input()

nimp=0
if nimp:
    print "biases"
    print biases[:3]
    print "1st 3 btstrp estimates of the mean"
    print cinq_cent_estimators_of_the_mean[:3]
    print "mean of observed data"
    print np.mean(X,axis=1)
    print "bias"
    print np.var(cinq_cent_estimators_of_the_mean)
    print "np.var(cinq_cent_estimators_of_the_mean)"

    #print np.sqrt(np.var(cinq_cent_estimators_of_the_mean,axis=1))
    #print "sqrt np.var(cinq_cent_estimators_of_the_mean)"
    print np.sqrt(np.var(cinq_cent_estimators_of_the_mean,axis=0))
    print "sqrt np.var(cinq_cent_estimators_of_the_mean) axis 0"
    print np.cov(np.transpose(cinq_cent_estimators_of_the_mean)).shape
    print "np.cov(np.transpose(cinq_cent_estimators_of_the_mean)).shape"
    print np.cov(np.transpose(cinq_cent_estimators_of_the_mean))
    print "the variances are given on the diagonal"
    #variance = 1/B


    #drop i-th observation from the sample


    print len(Xstar[0])
    print "Xstar"

    X_one=Xstar[0][:10]
    X_two=Xstar[1][:10]
    print X_one
    #raw_input()
    print  [X_one[a]   for a in range(len(X_one)) if a!=1]
    #del X_two[1]
    #print X_one

    #x[idx!=i]
    raw_input()

    #X_one=Xstar[0]
    #X_two=Xstar[1]
mean_jack_minus_mean_emp=[]
complete_bootstrap=[]
cinq_cent_estimators_of_the_mean_jack=[]
means_jack=[]
for b in range(B):
    
    # def jackknife(x, func):
    # """Jackknife estimate of the estimator func"""
    # n = len(x)
    # idx = np.arange(n)
    # return np.sum(func(x[idx!=i]) for i in range(n))/float(n)
    # #    print np.random.randint(n, size=3)
    #    Xstar_one_jack=[X_one[a]   for a in range(len(X_one)) if a!=b]
    #    Xstar_two_jack=[X_two[a]   for a in range(len(X_two)) if a!=b]
    # print len(X)
    # print "len(X)"
    # print len(X[0])
    # raw_input()
    Xstar_jack=[X[:,a]   for a in range(len(X[0])) if a!=b]
    complete_bootstrap.append(    Xstar_jack)
    #print Xstar_jack
    #    print Xstar_jack[0].shape
    
    # print "Xstar_jack[0].shape"
    # raw_input()
    
    #    print np.mean(Xstar_jack,axis=1).shape
    #    print "jackknife_mean=np.mean(Xstar_jack,axis=1)"
    #print np.mean(Xstar_jack,axis=0).shape
    #print "np.mean(Xstar_jack,axis=0)"
    mean_jack=np.mean(Xstar_jack,axis=0)
    # print mean_jack-np.mean(X,axis=1)
    # raw_input()

    #print" mean_jack-np.mean(X,axis=1)"
    #mean_jack_minus_mean_emp.append(mean_jack-np.mean(X,axis=1))
    means_jack.append(mean_jack)
    # print mean_jack_minus_mean_emp
    # raw_input()
useless=0
if useless:
    print np.mean(mean_jack_minus_mean_emp,axis=0)
    print "np.mean(mean_jack_minus_mean_emp,axis=0)"
    print (B-1)/B*np.mean(mean_jack_minus_mean_emp,axis=0)
    print "(B-1)/B*np.mean(mean_jack_minus_mean_emp)"
    print "(B-1)*np.cov(mean_jack_minus_mean_emp,bias=1) below"
    print (B-1)*np.cov(means_jack,bias=1)
    print np.cov(np.transpose(cinq_cent_estimators_of_the_mean))
    print "np.cov(np.transpose(cinq_cent_estimators_of_the_mean))"
    print len(cinq_cent_estimators_of_the_mean)
    print len(cinq_cent_estimators_of_the_mean[0])
    print len(means_jack)
    print len(means_jack[0])
    print cinq_cent_estimators_of_the_mean[0]
    print means_jack[0]
    print "cinq_cent_estimators_of_the_mean.shape"
    print np.cov(np.transpose(cinq_cent_estimators_of_the_mean),bias=1)
    print (B-1)*np.cov(np.transpose(means_jack),bias=1)
    print "(B-1)*np.cov(np.transpose(means_jack),bias=1)"
    #print np.cov(np.transpose(X),bias=1)/B
    print np.cov(X)/B
    print X.shape
    print len(means_jack)
    print len(means_jack[0])

    print "np.cov(np.transpose(X),bias=1)/B"
    raw_input()
    
    print "lalalilaloum"
    raw_input()
blou=0
if blou:
    print len(Xstar_jack)
    print "len(Xstar_jack)"
    raw_input()
    Xstar=X[:,np.random.randint(n, size=n)]
    print Xstar.shape
    print len(Xstar_jack)
    #    print len(Xstar_jack[0])
    print np.mean(Xstar_jack,axis=1)
    #print 'np.mean(Xstar,axis=1)'
    #print np.mean(Xstar_jack,axis=1)
    print "np.mean(Xstar_jack,axis=1)"
    raw_input()
    #    cinq_cent_estimators_of_the_mean_jack.append(np.mean(Xstar_jack,axis=1))


    #print     X[:,np.random.randint(n, size=4)]
    #raw_input()
    #    Xstarbarmem[:, b] = np.mean(Xstar,axis=1)

new_size=300
X1=np.random.uniform(low=0.0, high=1.0, size=new_size)
U=np.random.uniform(low=-0.1, high=0.1, size=new_size)
X2=X1+U
X_uni_matrix=np.vstack((X1, X2))
shapes_X_Q8=0
if shapes_X_Q8:
    print X1.shape
    print X2.shape
    print np.concatenate((X1, X2), axis=0).shape
    #print np.concatenate((X1, X2.T), axis=1).shape
    #print np.concatenate((X1.T, X2), axis=1).shape
    print np.vstack((X1, X2)).shape
    print np.hstack((X1, X2)).shape
    #print X1
    print np.corrcoef(X1,X2)
    print X1.shape
    print "X1.shape lala"
estimated_corr_coef_empiric=np.corrcoef(X1,X2)[0][1]


print "confidence intervals"
n=new_size
B=500
#Xstarbarmem = np.zeros((B,1))
cinq_cent_bootstrap_estimators_of_the_corr_coef=[]
for b in range(B):
    #    print np.random.randint(n, size=3)

    # the bootstrap samples
    Xstar=X_uni_matrix[:,np.random.randint(n, size=n)]

    #Xstar=X[:,np.random.randint(n, size=n)]
    #print Xstar.shape
    #print np.mean(Xstar,axis=1)
    #print 'np.mean(Xstar,axis=1)'
    X1star=Xstar[0,:]
    X2star=Xstar[1,:]
    #    print X1star.shape
    #    print "X1star.shape"
    #    raw_input()
    # print np.corrcoef(X1star,X2star)[0][1]
    # print "np.corrcoef(X1star,X2star)"
    # raw_input()
    
    cinq_cent_bootstrap_estimators_of_the_corr_coef.append(np.corrcoef(X1star,X2star)[0][1])
    #    cinq_cent_estimators_of_the_mean.append(np.mean(Xstar,axis=1))

graphe_hist_bootstrap_crr_coef_q8_exo2=0

if graphe_hist_bootstrap_crr_coef_q8_exo2:
    fig = plt.figure(figsize=(8, 6))
    plt.hist(cinq_cent_bootstrap_estimators_of_the_corr_coef, bins=25, normed=True, align='mid')

    #sns.kdeplot(cinq_cent_samples, shade=True, color="b")

    #sns.kdeplot(panda_array_meres, shade=True, color='#9932cc')

    plt.title(' Histogramme/ KDE : 500 samples of bootstrap replicas of corr coef. ' )
    ax = plt.gca()
    ax.legend_ = None
    plt.xlabel('Valeurs '), plt.ylabel('Frequences')
    plt.tight_layout()
    plt.show()
alpha=5 # ~ 5%  en pourcent!!!

q=[alpha/2.,100-alpha/2.]
print q #quantiles!

print "q"
raw_input()
print np.corrcoef(X1,X2)[0][1]
print "np.corrcoef(X1,X2)"

#estimated_corr_coef_empiric=np.corrcoef(X1,X2)[0][1]
print type(cinq_cent_bootstrap_estimators_of_the_corr_coef)
#print len(cinq_cent_bootstrap_estimators_of_the_corr_coef)
#print len(cinq_cent_bootstrap_estimators_of_the_corr_coef[0])
corr_coef_bootstrapp_minus_estimated=cinq_cent_bootstrap_estimators_of_the_corr_coef-estimated_corr_coef_empiric
#print len(corr_coef_bootstrapp_minus_estimated)
#print len(corr_coef_bootstrapp_minus_estimated[0])

graphe_hist_centered_bootstrap_crr_coef_q8_exo2=1
if graphe_hist_centered_bootstrap_crr_coef_q8_exo2:
    fig = plt.figure(figsize=(8, 6))
    plt.hist(corr_coef_bootstrapp_minus_estimated, bins=25, normed=True, align='mid')

    #sns.kdeplot(cinq_cent_samples, shade=True, color="b")

    #sns.kdeplot(panda_array_meres, shade=True, color='#9932cc')

    plt.title(' Histogramme/ KDE : 500 samples of bootstrap replicas of corr coef. -centered ' )
    ax = plt.gca()
    ax.legend_ = None
    plt.xlabel('Valeurs '), plt.ylabel('Frequences')
    plt.tight_layout()
    plt.show()
corr_coef_bootstrapp_minus_estimated_renorm_by_sqrt_n=np.sqrt(new_size)*(cinq_cent_bootstrap_estimators_of_the_corr_coef-estimated_corr_coef_empiric)

graphe_hist_centered_renorm_sqrt_n_bootstrap_crr_coef_q8_exo2=1
if graphe_hist_centered_renorm_sqrt_n_bootstrap_crr_coef_q8_exo2:
    
    fig = plt.figure(figsize=(8, 6))
    plt.hist(corr_coef_bootstrapp_minus_estimated_renorm_by_sqrt_n, bins=25, normed=True, align='mid')

    #sns.kdeplot(cinq_cent_samples, shade=True, color="b")

    #sns.kdeplot(panda_array_meres, shade=True, color='#9932cc')

    plt.title(' Histogramme/ KDE : 500 samples of bootstrap replicas of corr coef. -centered, renorm by sqrt(n) ' )
    ax = plt.gca()
    ax.legend_ = None
    plt.xlabel('Valeurs '), plt.ylabel('Frequences')
    plt.tight_layout()
    plt.show()


#print np.percentile(cinq_cent_bootstrap_estimators_of_the_corr_coef,q)
print np.percentile(corr_coef_bootstrapp_minus_estimated_renorm_by_sqrt_n,q)
lower_xi,upper_xi=np.percentile(corr_coef_bootstrapp_minus_estimated_renorm_by_sqrt_n,q)
print estimated_corr_coef_empiric+lower_xi/np.sqrt(n)
print estimated_corr_coef_empiric
print estimated_corr_coef_empiric+upper_xi/np.sqrt(n)


print ("xi le quantile vaut sqrt(2) erf^(-1)(1-alpha)")
alpha=5/100.0
import scipy

xi=np.sqrt(2)*scipy.special.erfinv(1-alpha)
print xi
