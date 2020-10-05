# coding=utf-8

"""
Goal: Generating fictive stock market curves for testing purposes.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import numpy as np
import pandas as pd
from scipy import signal
from dataDownloader import YahooFinance



###############################################################################
################################ Global variables #############################
###############################################################################

# Default values for the generation of the fictive stock market curves
MIN = 100
MAX = 200
PERIOD = 252



###############################################################################
############################ Class StockGenerator #############################
###############################################################################

class StockGenerator:
    """
    GOAL: Generation of some fictive stock market curves
          (linear, sinusoidal, triangle, etc.).
        
    VARIABLES: /
          
    METHODS: - linearUp: Generate a continuously increasing linear curve.
             - linearDown: Generate a continuously decreasing linear curve.
             - sinusoidal: Generate a (periodic) sinusoidal signal curve.
             - triangle: Generate a (periodic) triangle signal curve.
    """

    def linearUp (self, startingDate, endingDate, minValue=MIN, maxValue=MAX):
        """
        GOAL: Generate a new fictive stock market as a continuously increasing
              linear curve.
        
        INPUTS: - startingDate: Beginning of the trading horizon.  
                - endingDate: Ending of the trading horizon.
                - minValue: Minimum price value.   
                - maxValue: Maximum price value.    
        
        OUTPUTS: - linearUpward: Generated fictive stock market dataframe.
        """

        # Initialization of the new stock market dataframe
        downloader = YahooFinance()
        DowJones = downloader.getDailyData('DIA', startingDate, endingDate)
        linearUpward = pd.DataFrame(index=DowJones.index)

        # Generation of the fictive prices over the trading horizon
        length = len(linearUpward.index)
        prices = np.linspace(minValue, maxValue, num=length)

        # Filling of the new fictive stock market dataframe
        linearUpward['Open'] = prices
        linearUpward['High'] = prices 
        linearUpward['Low'] = prices
        linearUpward['Close'] = prices 
        linearUpward['Volume'] = 100000

        return linearUpward

    
    def linearDown (self, startingDate, endingDate, minValue=MIN, maxValue=MAX):
        """
        GOAL: Generate a new fictive stock market as a continuously decreasing
              linear curve.
        
        INPUTS: - startingDate: Beginning of the trading horizon.  
                - endingDate: Ending of the trading horizon.
                - minValue: Minimum price value.   
                - maxValue: Maximum price value.    
        
        OUTPUTS: - linearDownward: Generated fictive stock market dataframe.
        """

        # Initialization of the new stock market dataframe
        downloader = YahooFinance()
        DowJones = downloader.getDailyData('DIA', startingDate, endingDate)
        linearDownward = pd.DataFrame(index=DowJones.index)

        # Generation of the fictive prices over the trading horizon
        length = len(linearDownward.index)
        prices = np.linspace(minValue, maxValue, num=length)
        prices = np.flip(prices)

        # Filling of the new fictive stock market dataframe
        linearDownward['Open'] = prices
        linearDownward['High'] = prices 
        linearDownward['Low'] = prices
        linearDownward['Close'] = prices 
        linearDownward['Volume'] = 100000

        return linearDownward

    
    def sinusoidal(self, startingDate, endingDate, minValue=MIN, maxValue=MAX, period=PERIOD):
        """
        GOAL: Generate a new fictive stock market as a sinusoidal signal curve.
        
        INPUTS: - startingDate: Beginning of the trading horizon.  
                - endingDate: Ending of the trading horizon.
                - minValue: Minimum price value.   
                - maxValue: Maximum price value.  
                - period: Period of the sinusoidal signal.  
        
        OUTPUTS: - sinusoidal: Generated fictive stock market dataframe.
        """

        # Initialization of the new stock market dataframe
        downloader = YahooFinance()
        DowJones = downloader.getDailyData('DIA', startingDate, endingDate)
        sinusoidal = pd.DataFrame(index=DowJones.index)

        # Generation of the fictive prices over the trading horizon
        length = len(sinusoidal.index)
        t = np.linspace(0, length, num=length)
        prices = minValue + maxValue / 2 * (np.sin(2 * np.pi * t / period) + 1) / 2

        # Filling of the new fictive stock market dataframe
        sinusoidal['Open'] = prices
        sinusoidal['High'] = prices 
        sinusoidal['Low'] = prices
        sinusoidal['Close'] = prices 
        sinusoidal['Volume'] = 100000

        return sinusoidal

    
    def triangle(self, startingDate, endingDate, minValue=MIN, maxValue=MAX, period=PERIOD):
        """
        GOAL: Generate a new fictive stock market as a triangle signal curve.
        
        INPUTS: - startingDate: Beginning of the trading horizon.  
                - endingDate: Ending of the trading horizon.
                - minValue: Minimum price value.   
                - maxValue: Maximum price value.  
                - period: Period of the triangle signal.  
        
        OUTPUTS: - triangle: Generated fictive stock market dataframe.
        """

        # Initialization of the new stock market dataframe
        downloader = YahooFinance()
        DowJones = downloader.getDailyData('DIA', startingDate, endingDate)
        triangle = pd.DataFrame(index=DowJones.index)

        # Generation of the fictive prices over the trading horizon
        length = len(triangle.index)
        t = np.linspace(0, length, num=length)
        prices = minValue + maxValue / 2 * np.abs(signal.sawtooth(2 * np.pi * t / period))

        # Filling of the new fictive stock market dataframe
        triangle['Open'] = prices
        triangle['High'] = prices 
        triangle['Low'] = prices
        triangle['Close'] = prices 
        triangle['Volume'] = 100000

        return triangle
        