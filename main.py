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

if (__name__ == '__main__'):

    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-strategy",
                        default='TDQN',
                        type=str,
                        help="Name of the trading strategy")
    parser.add_argument("-stock",
                        default='Apple',
                        type=str,
                        help="Name of the stock (market)")
    parser.add_argument("-num_layers", default=5, type=int)
    parser.add_argument("-block_type",
                        default='linear',
                        type=str,
                        choices=['conv', 'linear', 'attention'])
    parser.add_argument("-num_episodes", default=50, type=int)
    args = parser.parse_args()

    # Initialization of the required variables
    simulator = TradingSimulator()
    strategy = args.strategy
    stock = args.stock

    kwargs = dict(
        numberOfLayers=args.num_layers,
        blockType=args.block_type,
    )
    # Training and testing of the trading strategy specified for the stock (market) specified
    simulator.simulateNewStrategy(strategy,
                                  stock,
                                  saveStrategy=False,
                                  numberOfEpisodes=args.num_episodes,
                                  **kwargs)
    """
    simulator.displayTestbench()
    simulator.analyseTimeSeries(stock)
    simulator.simulateNewStrategy(strategy, stock, saveStrategy=False, **kwargs)
    simulator.simulateExistingStrategy(strategy, stock, **kwargs)
    simulator.evaluateStrategy(strategy, saveStrategy=False, numberOfEpisodes=args.num_episodes, **kwargs)
    simulator.evaluateStock(stock, numberOfEpisodes=args.num_episodes, **kwargs)
    """
