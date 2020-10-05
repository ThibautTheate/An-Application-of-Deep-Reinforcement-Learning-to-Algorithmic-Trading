# coding=utf-8

"""
Goal: Accurately estimating the performance of a trading strategy.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import numpy as np

from tabulate import tabulate
from matplotlib import pyplot as plt



###############################################################################
######################### Class PerformanceEstimator ##########################
###############################################################################

class PerformanceEstimator:
    """
    GOAL: Accurately estimating the performance of a trading strategy, by
          computing many different performance indicators.
        
    VARIABLES: - data: Trading activity data from the trading environment.
               - PnL: Profit & Loss (performance indicator).
               - annualizedReturn: Annualized Return (performance indicator).
               - annualizedVolatily: Annualized Volatility (performance indicator).
               - profitability: Profitability (performance indicator).
               - averageProfitLossRatio: Average Profit/Loss Ratio (performance indicator).
               - sharpeRatio: Sharpe Ratio (performance indicator).
               - sortinoRatio: Sortino Ratio (performance indicator).
               - maxDD: Maximum Drawdown (performance indicator).
               - maxDDD: Maximum Drawdown Duration (performance indicator).
               - skewness: Skewness of the returns (performance indicator).
          
    METHODS:   -  __init__: Object constructor initializing some class variables. 
               - computePnL: Compute the P&L.
               - computeAnnualizedReturn: Compute the Annualized Return.
               - computeAnnualizedVolatility: Compute the Annualized Volatility.
               - computeProfitability: Computate both the Profitability and the Average Profit/Loss Ratio.
               - computeSharpeRatio: Compute the Sharpe Ratio.
               - computeSortinoRatio: Compute the Sortino Ratio.
               - computeMaxDrawdown: Compute both the Maximum Drawdown and Maximum Drawdown Duration.
               - computeSkewness: Compute the Skewness of the returns.
               - computePerformance: Compute all the performance indicators.
               - displayPerformance: Display the entire set of performance indicators in a table.
    """

    def __init__(self, tradingData):
        """
        GOAL: Object constructor initializing the class variables. 
        
        INPUTS: - tradingData: Trading data from the trading strategy execution.
        
        OUTPUTS: /
        """

        self.data = tradingData


    def computePnL(self):
        """
        GOAL: Compute the Profit & Loss (P&L) performance indicator, which
              quantifies the money gained or lost during the trading activity.
        
        INPUTS: /
        
        OUTPUTS:    - PnL: Profit or loss (P&L) performance indicator.
        """
        
        # Compute the PnL
        self.PnL = self.data["Money"][-1] - self.data["Money"][0]
        return self.PnL
    

    def computeAnnualizedReturn(self):
        """
        GOAL: Compute the yearly average profit or loss (in %), called
              the Annualized Return performance indicator.
        
        INPUTS: /
        
        OUTPUTS:    - annualizedReturn: Annualized Return performance indicator.
        """
        
        # Compute the cumulative return over the entire trading horizon
        cumulativeReturn = self.data['Returns'].cumsum()
        cumulativeReturn = cumulativeReturn[-1]
        
        # Compute the time elapsed (in days)
        start = self.data.index[0].to_pydatetime()
        end = self.data.index[-1].to_pydatetime()     
        timeElapsed = end - start
        timeElapsed = timeElapsed.days

        # Compute the Annualized Return
        if(cumulativeReturn > -1):
            self.annualizedReturn = 100 * (((1 + cumulativeReturn) ** (365/timeElapsed)) - 1)
        else:
            self.annualizedReturn = -100
        return self.annualizedReturn
    
    
    def computeAnnualizedVolatility(self):
        """
        GOAL: Compute the Yearly Voltility of the returns (in %), which is
              a measurement of the risk associated with the trading activity.
        
        INPUTS: /
        
        OUTPUTS:    - annualizedVolatily: Annualized Volatility performance indicator.
        """
        
        # Compute the Annualized Volatility (252 trading days in 1 trading year)
        self.annualizedVolatily = 100 * np.sqrt(252) * self.data['Returns'].std()
        return self.annualizedVolatily
    
    
    def computeSharpeRatio(self, riskFreeRate=0):
        """
        GOAL: Compute the Sharpe Ratio of the trading activity, which is one of
              the most suited performance indicator as it balances the brute
              performance and the risk associated with a trading activity.
        
        INPUTS:     - riskFreeRate: Return of an investment with a risk null.
        
        OUTPUTS:    - sharpeRatio: Sharpe Ratio performance indicator.
        """
        
        # Compute the expected return
        expectedReturn = self.data['Returns'].mean()
        
        # Compute the returns volatility
        volatility = self.data['Returns'].std()
        
        # Compute the Sharpe Ratio (252 trading days in 1 year)
        if expectedReturn != 0 and volatility != 0:
            self.sharpeRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
        else:
            self.sharpeRatio = 0
        return self.sharpeRatio
    
    
    def computeSortinoRatio(self, riskFreeRate=0):
        """
        GOAL: Compute the Sortino Ratio of the trading activity, which is similar
              to the Sharpe Ratio but does no longer penalize positive risk.
        
        INPUTS:     - riskFreeRate: Return of an investment with a risk null.
        
        OUTPUTS:    - sortinoRatio: Sortino Ratio performance indicator.
        """
        
        # Compute the expected return
        expectedReturn = np.mean(self.data['Returns'])
        
        # Compute the negative returns volatility
        negativeReturns = [returns for returns in self.data['Returns'] if returns < 0]
        volatility = np.std(negativeReturns)
        
        # Compute the Sortino Ratio (252 trading days in 1 year)
        if expectedReturn != 0 and volatility != 0:
            self.sortinoRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
        else:
            self.sortinoRatio = 0
        return self.sortinoRatio
    
    
    def computeMaxDrawdown(self, plotting=False):
        """
        GOAL: Compute both the Maximum Drawdown and the Maximum Drawdown Duration
              performance indicators of the trading activity, which are measurements
              of the risk associated with the trading activity.
        
        INPUTS: - plotting: Boolean enabling the maximum drawdown plotting.
        
        OUTPUTS:    - maxDD: Maximum Drawdown performance indicator.
                    - maxDDD: Maximum Drawdown Duration performance indicator.
        """

        # Compute both the Maximum Drawdown and Maximum Drawdown Duration
        capital = self.data['Money'].values
        through = np.argmax(np.maximum.accumulate(capital) - capital)
        if through != 0:
            peak = np.argmax(capital[:through])
            self.maxDD = 100 * (capital[peak] - capital[through])/capital[peak]
            self.maxDDD = through - peak
        else:
            self.maxDD = 0
            self.maxDDD = 0
            return self.maxDD, self.maxDDD

        # Plotting of the Maximum Drawdown if required
        if plotting:
            plt.figure(figsize=(10, 4))
            plt.plot(self.data['Money'], lw=2, color='Blue')
            plt.plot([self.data.iloc[[peak]].index, self.data.iloc[[through]].index],
                     [capital[peak], capital[through]], 'o', color='Red', markersize=5)
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.savefig(''.join(['Figures/', 'MaximumDrawDown', '.png']))
            #plt.show()

        # Return of the results
        return self.maxDD, self.maxDDD
    

    def computeProfitability(self):
        """
        GOAL: Compute both the percentage of trades that resulted
              in profit (Profitability), and the ratio between the
              average profit and the average loss (AverageProfitLossRatio).
        
        INPUTS: /
        
        OUTPUTS:    - profitability: Percentage of trades that resulted in profit.
                    - averageProfitLossRatio: Ratio between the average profit
                                              and the average loss.
        """
        
        # Initialization of some variables
        good = 0
        bad = 0
        profit = 0
        loss = 0
        index = next((i for i in range(len(self.data.index)) if self.data['Action'][i] != 0), None)
        if index == None:
            self.profitability = 0
            self.averageProfitLossRatio = 0
            return self.profitability, self.averageProfitLossRatio
        money = self.data['Money'][index]

        # Monitor the success of each trade over the entire trading horizon
        for i in range(index+1, len(self.data.index)):
            if(self.data['Action'][i] != 0):
                delta = self.data['Money'][i] - money
                money = self.data['Money'][i]
                if(delta >= 0):
                    good += 1
                    profit += delta
                else:
                    bad += 1
                    loss -= delta

        # Special case of the termination trade
        delta = self.data['Money'][-1] - money
        if(delta >= 0):
            good += 1
            profit += delta
        else:
            bad += 1
            loss -= delta

        # Compute the Profitability
        self.profitability = 100 * good/(good + bad)
         
        # Compute the ratio average Profit/Loss  
        if(good != 0):
            profit /= good
        if(bad != 0):
            loss /= bad
        if(loss != 0):
            self.averageProfitLossRatio = profit/loss
        else:
            self.averageProfitLossRatio = float('Inf')

        return self.profitability, self.averageProfitLossRatio
        

    def computeSkewness(self):
        """
        GOAL: Compute the skewness of the returns, which is
              a measurement of the degree of distorsion
              from the symmetrical bell curve.
        
        INPUTS: /
        
        OUTPUTS:    - skewness: Skewness performance indicator.
        """
        
        # Compute the Skewness of the returns
        self.skewness = self.data["Returns"].skew()
        return self.skewness
        
    
    def computePerformance(self):
        """
        GOAL: Compute the entire set of performance indicators.
        
        INPUTS: /
        
        OUTPUTS:    - performanceTable: Table summarizing the performance of 
                                        a trading strategy.
        """
    
        # Compute the entire set of performance indicators
        self.computePnL()
        self.computeAnnualizedReturn()
        self.computeAnnualizedVolatility()
        self.computeProfitability()
        self.computeSharpeRatio()
        self.computeSortinoRatio()
        self.computeMaxDrawdown()
        self.computeSkewness()

        # Generate the performance table
        self.performanceTable = [["Profit & Loss (P&L)", "{0:.0f}".format(self.PnL)], 
                                 ["Annualized Return", "{0:.2f}".format(self.annualizedReturn) + '%'],
                                 ["Annualized Volatility", "{0:.2f}".format(self.annualizedVolatily) + '%'],
                                 ["Sharpe Ratio", "{0:.3f}".format(self.sharpeRatio)],
                                 ["Sortino Ratio", "{0:.3f}".format(self.sortinoRatio)],
                                 ["Maximum Drawdown", "{0:.2f}".format(self.maxDD) + '%'],
                                 ["Maximum Drawdown Duration", "{0:.0f}".format(self.maxDDD) + ' days'],
                                 ["Profitability", "{0:.2f}".format(self.profitability) + '%'],
                                 ["Ratio Average Profit/Loss", "{0:.3f}".format(self.averageProfitLossRatio)],
                                 ["Skewness", "{0:.3f}".format(self.skewness)]]
        
        return self.performanceTable


    def displayPerformance(self, name):
        """
        GOAL: Compute and display the entire set of performance indicators
              in a table.
        
        INPUTS: - name: Name of the element (strategy or stock) analysed.
        
        OUTPUTS:    - performanceTable: Table summarizing the performance of 
                                        a trading activity.
        """
        
        # Generation of the performance table
        self.computePerformance()
        
        # Display the table in the console (Tabulate for the beauty of the print operation)
        headers = ["Performance Indicator", name]
        tabulation = tabulate(self.performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        print(tabulation)
    