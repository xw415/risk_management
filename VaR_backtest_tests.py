import sys
#!{sys.executable} -m pip install hurst
import pickle
import math
import numpy as np
from collections import Counter
from hurst import compute_Hc
from IPython.display import display
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import kstest, uniform
from scipy.stats import expon

from quantile_functions import *

def exceptions_by_ticker(return_set, look_back, divider, c):
    """count  the number of out-sample exceptions based on model coefficients
    input: 
    return_set = data frame of returns
    look_back = the number of lagged squared returns
    divider = the year separating the training vs the test data
    c = c[0], coefficients for the lagged squared returns and c[1], the intercept
    return:
    the percent of days that are receptions
    the number of rows and columns in the test data set
    1-0 vector of exceptions
    """
    r_history, _ = create_lagged_returns(return_set, look_back)
    
    _, variance_test, _, _ = get_historical_variances(r_history, divider)  ##############   need to generalize this beyond variance  ##########
    
    quantile_prediction = variance_test.iloc[:,1:].dot(c[0]) + c[1] * variance_test.shape[0]
    exceptions = [1 if r < p else 0 for r, p in zip(variance_test.iloc[:,0], quantile_prediction)]
    return sum(exceptions) / len(exceptions), variance_test.shape[0], exceptions

from scipy.stats import chi2, norm, expon

class VaR_goodness_of_fit():

    def __init__(self, returns, Value_at_Risk, p):
        self.returns = returns
        self.Value_at_Risk = Value_at_Risk
        self.e = returns < Value_at_Risk
        self.p = p                
        self.fcn_list = [
            self.Kupiec_uc_test,
            self.Kupiec_ind_test,
            self.Kupiec_tests,
            self.uniform_test,
            self.fit_to_expon,
        ]

    def do_Kupiec_uc_test(self, N, x, p):
        """Kupiec unit count test
        input: 
        N = the number of days
        x = the number of exceptions
        p = the target percentage of days that are exceptions
        """
        LRuc = -2 * np.log(pow(p, x) * pow(1 - p, N - x) / (pow(x / N, x) * pow(1 - x / N, N - x)))
        return {"LRuc": LRuc, "pvalue": chi2(1).cdf(LRuc)}  

    def Kupiec_uc_test(self):
        """call Kuprice unit count test
        input:
        e = 1-0 vector indicating days that are exceptions
        p = the target percentag of days that are exceptions
        """
        return self.do_Kupiec_uc_test(len(self.e), sum(self.e), self.p)

    def Kupiec_ind_test(self):
        """Kupiec independence test
        input:
        e = 1-0 vector indicting days that are exceptions
        """
        T = dict(Counter([(a,b) for a,b in zip(self.e[:-1], self.e[1:])]).most_common(4))
        if (1,1) not in T.keys():
            T[(1,1)] = 0
        
        PI_01 = T[(0,1)] / (T[(0,0)] + T[(0,1)])
        PI_11 = T[(1,1)] / (T[(1,0)] + T[(1,1)])
        PI = (T[(0,1)] + T[(1,1)]) / (len(self.e) - 1)

        LRind = 2 * math.log(
            pow(1 - PI_01, T[(0,0)]) * pow(PI_01, T[(0,1)]) * pow(1 - PI_11, T[(1,0)]) * pow(PI_11, T[(1,1)]) /
            (pow(1 - PI, T[(0,0)] + T[(1,0)]) * pow(PI, T[(0,1)] + T[(1,1)]))
        )
        return {"LRind": LRind, "pvalue": chi2(2).cdf(LRind)}

    def Kupiec_tests(self):
        """Kupiec test combining unit count and independence test
        input:
        e = 1-0 vector indicating days that are exceptions
        p = the target percentag of days that are exceptions
        """
        LR = self.Kupiec_uc_test()["LRuc"] + self.Kupiec_ind_test()["LRind"]
        return {"LR": LR, "pvalue": chi2(2).cdf(LR)}

    def get_z_score(self, p:float, n:int, actual:int):
        """binomial distribution z score and probability"""
        z = (actual - p) / (math.sqrt(p * (1-p) / n))
        return z, st.norm.cdf(z)

    def binomial_PF_test(self, n, x, p):
        """binomial distribution z score and probability"""
        z = (p - x / n) / math.sqrt(p * (1-p) / n)
        return {"z": z, "pvalue": norm.cdf(z)}

    def uniform_test(self):
        """Kolmogorov Smirnov test if exceptions are uniformly distributed overtime
        input:
        e = 1-0 vector indicating days that are exceptions
        """
        e1 = (np.array([i for i,e in enumerate(self.e) if e > 0]) / len(self.e))
        ks = kstest(e1, 'uniform')
        return {"uniform distribution statistic": ks.statistic, "pvalue": ks.pvalue}

    # https://en.wikipedia.org/wiki/Exponential_distribution

    def fit_to_expon(self):
        gaps = np.diff(np.array([i for i,x in enumerate(self.e) if x ==1]))
        ks = kstest(gaps, 'expon', args=([0, 1 / self.p]))
        return {"interval statistic": ks.statistic, "pvalue": ks.pvalue}

    def reformat_results(self, f):
        f_values = f()
        keys = list(f_values.keys())
        keys.remove('pvalue')
        return (keys[0], f_values[keys[0]], f_values['pvalue'])

    def all_test_results(self):
        test_results0 = pd.DataFrame([self.reformat_results(f) for f in self.fcn_list], columns=['test','statistic','pvalue'])
        test_results = test_results0[['statistic','pvalue']]
        test_results.index = test_results0['test']
        return test_results

    def print_exception_pct(self):
        exception_count = sum(self.e)
        observations = len(self.Value_at_Risk)
        print(f'exceptions = {exception_count} percent = {exception_count / observations: 0.4} expected = {self.p: 0.4}')

    def scatterplot_returns_vs_VaR(self):
        colors = np.array([[255,0,0] if e else [0,0,255] for e in self.e])
        plt.scatter(self.returns , self.Value_at_Risk, c=colors / 255)
        plt.xlabel('returns')
        plt.ylabel('value at Risk')
        plt.show()
    

def compute_autocorrelation(x, n=20):
    c = [np.corrcoef(x[t:], x[:-t])[0][1] for t in range(1, n)]
    # reverse series, add a 1.0, and add the original series
    c_auto = c[::-1]
    c_auto.append(1.0)
    c_auto.extend(c)
    return c_auto

def plot_autocorrelation(x, title, n=20):
    c_auto = compute_autocorrelation(x, n)

    plt.plot(np.arange(-n + 1, n), c_auto, 'o-')
    plt.plot([-n, n], [0, 0], 'r--')
    plt.title(f'{title} autocorrelation')
    plt.xlabel('offset')
    plt.ylabel('correlation')
    plt.show()
    
'''
    hist = plt.hist(gaps,40)
    plt.clf()

    plt.bar(hist[1][:-1], hist[0]/sum(hist[0]),5)
    x = np.linspace(expon.ppf(0.),max(hist[1]), 100)
    plt.plot(x, 5 * expon.pdf(x, scale=10),'r-', lw=5, alpha=.6, label='expon pdf')
    plt.show()
    '''
