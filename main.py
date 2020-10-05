# coding=utf-8

"""
Goal: Program Main.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import argparse

from tradingSimulator import TradingSimulator



###############################################################################
##################################### MAIN ####################################
###############################################################################

if(__name__ == '__main__'):

    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-strategy", default='TDQN', type=str, help="Name of the trading strategy")
    parser.add_argument("-stock", default='Apple', type=str, help="Name of the stock (market)")
    args = parser.parse_args()
    
    # Initialization of the required variables
    simulator = TradingSimulator()
    strategy = args.strategy
    stock = args.stock

    # Training and testing of the trading strategy specified for the stock (market) specified
    simulator.simulateNewStrategy(strategy, stock, saveStrategy=True)
    """
    simulator.simulateNewStrategy(strategy, stock, saveStrategy=True)
    simulator.displayTestbench()
    simulator.analyseTimeSeries(stock)
    simulator.simulateExistingStrategy(strategy, stock)
    simulator.evaluateStock(stock)
    """
