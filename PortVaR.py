import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data
from scipy.stats import norm, t, kurtosis, skew

#To define the number of observations
enddate = dt.datetime.now()
startdate = enddate - dt.timedelta(days=520)

#To define the stocklist
stocklist = ['AAPL', 'CAT', 'GLD', 'SLV', 'BAC', 'NVDA', 'KO', 'USO', 'MCD']
stocks = [stock for stock in stocklist]

#To import the data from yahoo finance
stockdata = data.DataReader(stocks, data_source='yahoo', start=startdate)
stockdata = stockdata["Close"]
logreturns = np.log(1 + stockdata.pct_change()) #To calculate log returns from normal percentage returns.
meanreturns = logreturns.mean()
covmatrix = logreturns.cov() #To calculate variance-covariance matrix from log returns.

#To define weights randomly for the number of stocks printed above in "meanreturns".
weights = np.random.random(len(meanreturns))
#The operator "/=" is short to calculate proportions, ie, "x=x/n". This operator naturaly calculates proportions to sum 100%.
weights /= np.sum(weights)
#To transform a numpy array into a pandas dataframe to output them in the excel file.
weights_output = pd.DataFrame({'Symbols': stocklist, 'weights': weights})

#To multiply the two arrays meanreturns transposed and stock weights to get the portfolio mean return.
port_mean_return = np.dot(meanreturns.T, weights) #For operations about arrays and matrices: https://problemsolvingwithpython.com/05-NumPy-and-Arrays/05.07-Array-Opperations/

#The square root of multiplication of the stock weights array transposed by the covariance matrix and again by the weights array to get the portfolio volatility.
port_vol = np.sqrt(np.dot(np.dot(weights.T, covmatrix), weights))

#To get the probability density function of 99% confidence level. https://www.youtube.com/watch?v=bTZP86fHIuI
alpha = 0.01
Znorm = norm.ppf(1 - alpha)
Ztstudent = t.ppf(1 - alpha, len(logreturns) - 1) #Second part is for the degrees of freedom (n-1).
kurtosis = (kurtosis(meanreturns, fisher = False))
skewness = (skew(meanreturns))

investment = 1e6

#To calculate VaR
#If kurtosis is below the value 3, then it has fatter tails and we should use t-student.
if kurtosis < 3:
	VaR_1day = (port_mean_return + Ztstudent * port_vol) * investment
else:
	VaR_1day = (port_mean_return + Znorm * port_vol) * investment

VaR_10day = VaR_1day * np.sqrt(10)

#To transform the several numpy scalars to a single dataframe. To display the scalars we need to use [].
report = pd.DataFrame({'Portfolio mean return %': [port_mean_return * 100], 'Portfolio volatility %': [port_vol * 100], 'kurtosis': [kurtosis], 'skewness': [skewness], '1 day VaR': [VaR_1day], '10 day VaR': [VaR_10day],})

print('\nPortfolio mean return:    % '+ str(port_mean_return * 100))
print('Portfolio volatility:     % '+ str(port_vol * 100))
print('kurtosis:                   '+ str(kurtosis))
print('skewness:                   '+ str(skewness))
print('1 day VaR                 $ '+ str(VaR_1day))
print('10 day VaR                $ '+ str(VaR_10day))

#To produce an output file.
with pd.ExcelWriter('output_portvar.xlsx') as writer:
	stockdata.to_excel(writer, sheet_name = 'prices')
	logreturns.to_excel(writer, sheet_name = 'returns')
	covmatrix.to_excel(writer, sheet_name = 'covmatrix')
	meanreturns.to_excel(writer, sheet_name = 'meanreturns')
	weights_output.to_excel(writer, sheet_name = 'weights')
	report.to_excel(writer, sheet_name = 'report')
