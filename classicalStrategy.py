# coding=utf-8

"""
Goal: Implementing some classical algorithmic trading strategies.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import math
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod

from tradingPerformance import PerformanceEstimator



###############################################################################
########################### Class tradingStrategy #############################
###############################################################################

class tradingStrategy(ABC):
    """
    GOAL: Define the abstract class representing a classical trading strategy.
        
    VARIABLES: /
          
    METHODS: - chooseAction: Make a decision regarding the next trading
                             position (long=1 and short=0).
             - training: Train the trading strategy on a known trading
                         environment (called training set) in order to
                         tune the trading strategy parameters.
             - testing: Test the trading strategy on another unknown trading
                        environment (called testing set) in order to evaluate
                        the trading strategy performance.
    """

    @abstractmethod
    def chooseAction(self, state):
        """
        GOAL: Make a decision regarding the next trading position
              (long=1 and short=0).
        
        INPUTS: - state: State of the trading environment.      
        
        OUTPUTS: - action: Trading position decision (long=1 and short=0).
        """

        pass
    

    @abstractmethod
    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        """
        GOAL: Train the trading strategy on a known trading environment
              (called training set) in order to tune the trading strategy
              parameters.
        
        INPUTS: - trainingEnv: Known trading environment (training set).
                - trainingParameters: Additional parameters associated
                                      with the training phase.   
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the trading environment rendering.
                - plotTraining: Enable the plotting of the training results.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - trainingEnv: Trading environment associated with the best
                                trading strategy parameters backtested.
        """

        pass


    @abstractmethod
    def testing(self, testingEnv, trainingEnv, rendering=False, showPerformance=False):
        """
        GOAL: Test the trading strategy on another unknown trading
              environment (called testing set) in order to evaluate
              the trading strategy performance.
        
        INPUTS: - testingEnv: Unknown trading environment (testing set).
                - trainingEnv: Known trading environment (training set).
                - rendering: Enable the trading environment rendering.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - testingEnv: Trading environment backtested.
        """

        pass



###############################################################################
############################## Class BuyAndHold ###############################
###############################################################################

class BuyAndHold(tradingStrategy):
    """
    GOAL: Implement a simple "Buy and Hold" trading strategy.
        
    VARIABLES: /
          
    METHODS: - chooseAction: Always make the long trading position decision
                             (return 1).
             - training: Train the trading strategy on a known trading
                         environment (called training set) in order to
                         tune the trading strategy parameters. However,
                         there is no training required for a simple
                         Buy and Hold strategy because the strategy does not
                         involve any tunable parameter.
             - testing: Test the trading strategy on another unknown trading
                        environment (called testing set) in order to evaluate
                        the trading strategy performance.
    """

    def chooseAction(self, state):
        """
        GOAL: Always make the long trading position decision.
        
        INPUTS: - state: State of the trading environment.      
        
        OUTPUTS: - action: Trading position decision (always long -> 1).
        """

        # Trading position decision -> always long
        return 1
    

    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        """
        GOAL: Train the trading strategy on a known trading environment
              (called training set) in order to tune the trading strategy
              parameters. However, there is no training required for a
              simple Buy and Hold strategy because the strategy does not
              involve any tunable parameter.
        
        INPUTS: - trainingEnv: Known trading environment (training set).
                - trainingParameters: Additional parameters associated
                                      with the training phase. None for
                                      the Buy and Hold strategy.  
                - verbose: Enable the printing of a training feedback. None
                           for the Buy and Hold strategy.
                - rendering: Enable the trading environment rendering.
                - plotTraining: Enable the plotting of the training results.
                                None for the Buy and Hold strategy.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - trainingEnv: Trading environment backtested.
        """

        # Execution of the trading strategy on the trading environment
        trainingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))

        # If required, print a feedback about the training
        if verbose:
            print("No training is required as the simple Buy and Hold trading strategy does not involve any tunable parameters.")
        
        # If required, render the trading environment backtested
        if rendering:
            trainingEnv.render()
        
        # If required, plot the training results
        if plotTraining:
            print("No training results are available as the simple Buy and Hold trading strategy does not involve any tunable parameters.")

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('B&H')

        # Return the trading environment backtested (training set)
        return trainingEnv


    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        """
        GOAL: Test the trading strategy on another unknown trading
              environment (called testing set) in order to evaluate
              the trading strategy performance.
        
        INPUTS: - testingEnv: Unknown trading environment (testing set).
                - trainingEnv: Known trading environment (training set).
                - rendering: Enable the trading environment rendering.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - testingEnv: Trading environment backtested.
        """

        # Execution of the trading strategy on the trading environment
        testingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = testingEnv.step(self.chooseAction(testingEnv.state))

        # If required, render the trading environment backtested
        if rendering:
            testingEnv.render()

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('B&H')

        # Return the trading environment backtested (testing set)
        return testingEnv



###############################################################################
############################## Class SellAndHold ###############################
###############################################################################

class SellAndHold(tradingStrategy):
    """
    GOAL: Implement a simple "Sell and Hold" trading strategy.
        
    VARIABLES: /
          
    METHODS: - chooseAction: Always make the short trading position decision
                             (return 0).
             - training: Train the trading strategy on a known trading
                         environment (called training set) in order to
                         tune the trading strategy parameters. However,
                         there is no training required for a simple
                         Sell and Hold strategy because the strategy does not
                         involve any tunable parameter.
             - testing: Test the trading strategy on another unknown trading
                        environment (called testing set) in order to evaluate
                        the trading strategy performance.
    """

    def chooseAction(self, state):
        """
        GOAL: Always make the short trading position decision.
        
        INPUTS: - state: State of the trading environment.      
        
        OUTPUTS: - action: Trading position decision (always short -> 0).
        """

        # Trading position decision -> always short
        return 0
    

    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        """
        GOAL: Train the trading strategy on a known trading environment
              (called training set) in order to tune the trading strategy
              parameters. However, there is no training required for a
              simple Sell and Hold strategy because the strategy does not
              involve any tunable parameter.
        
        INPUTS: - trainingEnv: Known trading environment (training set).
                - trainingParameters: Additional parameters associated
                                      with the training phase. None for
                                      the Sell and Hold strategy.  
                - verbose: Enable the printing of a training feedback. None
                           for the Sell and Hold strategy.
                - rendering: Enable the trading environment rendering.
                - plotTraining: Enable the plotting of the training results.
                                None for the Sell and Hold strategy.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - trainingEnv: Trading environment backtested.
        """

        # Execution of the trading strategy on the trading environment
        trainingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))

        # If required, print a feedback about the training
        if verbose:
            print("No training is required as the simple Sell and Hold trading strategy does not involve any tunable parameters.")
        
        # If required, render the trading environment backtested
        if rendering:
            trainingEnv.render()
        
        # If required, plot the training results
        if plotTraining:
            print("No training results are available as the simple Sell and Hold trading strategy does not involve any tunable parameters.")

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('S&H')

        # Return the trading environment backtested (training set)
        return trainingEnv


    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        """
        GOAL: Test the trading strategy on another unknown trading
              environment (called testing set) in order to evaluate
              the trading strategy performance.
        
        INPUTS: - testingEnv: Unknown trading environment (testing set).
                - trainingEnv: Known trading environment (training set).
                - rendering: Enable the trading environment rendering.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - testingEnv: Trading environment backtested.
        """

        # Execution of the trading strategy on the trading environment
        testingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = testingEnv.step(self.chooseAction(testingEnv.state))

        # If required, render the trading environment backtested
        if rendering:
            testingEnv.render()

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('S&H')

        # Return the trading environment backtested (testing set)
        return testingEnv



###############################################################################
############################ Class MovingAveragesTF ###########################
###############################################################################

class MovingAveragesTF(tradingStrategy):
    """
    GOAL: Implement a Trend Following trading strategy based on moving averages.
        
    VARIABLES: - parameters: Trading strategy parameters, which are the windows
                             durations of the moving averages.
          
    METHODS: - __init__: Object constructor initializing the strategy parameters.
             - setParameters: Set new values for the parameters of the trading
                              strategy, which are the two windows durations.
             - processState: Process the trading environment state to obtain 
                             the required format.
             - chooseAction: Make a decision regarding the next trading
                             position (long=1 and short=0).
             - training: Train the trading strategy on a known trading
                         environment (called training set) in order to
                         tune the trading strategy parameters.
             - testing: Test the trading strategy on another unknown trading
                        environment (called testing set) in order to evaluate
                        the trading strategy performance.
    """

    def __init__(self, parameters=[5, 10]):
        """
        GOAL: Object constructor initializing the strategy parameters.
        
        INPUTS: - parameters: Trading strategy parameters, which are the windows
                              durations of the moving averages. 
        
        OUTPUTS: /
        """

        self.parameters = parameters


    def setParameters(self, parameters):
        """
        GOAL: Set new values for the parameters of the trading strategy, 
              which are the two windows durations.
        
        INPUTS: - parameters: List of new parameters to set.      
        
        OUTPUTS: /
        """

        self.parameters = parameters

    
    def processState(self, state):
        """
        GOAL: Process the trading environment state to obtain the required format.
        
        INPUTS: - state: State of the trading environment.
        
        OUTPUTS: - state: State of the trading environment in the required format.
        """

        return state[0]


    def chooseAction(self, state):
        """
        GOAL: Make a decision regarding the next trading position
              (long=1 and short=0) based on the moving averages.
        
        INPUTS: - state: State of the trading environment.      
        
        OUTPUTS: - action: Trading position decision (long=1 and short=0).
        """

        # Processing of the trading environment state
        state = self.processState(state)

        # Computation of the two moving averages
        shortAverage = np.mean(state[-self.parameters[0]:])
        longAverage = np.mean(state[-self.parameters[1]:])

        # Comparison of the two moving averages
        if(shortAverage >= longAverage):
            # Long position
            return 1
        else:
            # Short position
            return 0
    

    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        """
        GOAL: Train the trading strategy on a known trading environment
              (called training set) in order to tune the trading strategy
              parameters, by simulating many combinations of parameters.
        
        INPUTS: - trainingEnv: Known trading environment (training set).
                - trainingParameters: Additional parameters associated
                                      with the training phase simulations.   
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the trading environment rendering.
                - plotTraining: Enable the plotting of the training results.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - trainingEnv: Trading environment associated with the best
                                trading strategy parameters backtested.
        """

        # Compute the dimension of the parameter search space
        bounds = trainingParameters[0]
        step = trainingParameters[1]
        dimension = math.ceil((bounds[1] - bounds[0])/step)

        # Initialize some variables required for the simulations
        trainingEnv.reset()
        results = np.zeros((dimension, dimension))
        bestShort = 0
        bestLong = 0
        bestPerformance = -100
        i = 0
        j = 0
        count = 1

        # If required, compute the number of simulation iterations
        if verbose:
            iterations = dimension - 1
            length = 0
            while iterations > 0:
                length += iterations
                iterations -= 1

        # Loop through all the parameters combinations included in the parameter search space
        for shorter in range(bounds[0], bounds[1], step):
            for longer in range(bounds[0], bounds[1], step):

                # Obvious restriction on the parameters
                if(shorter < longer):

                    # If required, print the progression of the training
                    if(verbose):       
                        print("".join(["Training progression: ", str(count), "/", str(length)]), end='\r', flush=True)

                    # Apply the trading strategy with the current combination of parameters
                    self.setParameters([shorter, longer])
                    done = 0
                    while done == 0:
                        _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))

                    # Retrieve the performance associated with this simulation (Sharpe Ratio)
                    performanceAnalysis = PerformanceEstimator(trainingEnv.data)
                    performance = performanceAnalysis.computeSharpeRatio()
                    results[i][j] = performance

                    # Track the best performance and parameters
                    if(performance > bestPerformance):
                        bestShort = shorter
                        bestLong = longer
                        bestPerformance = performance
                    
                    # Reset of the trading environment
                    trainingEnv.reset()
                    count += 1

                j += 1
            i += 1
            j = 0

        # Execute once again the strategy associated with the best parameters simulated
        trainingEnv.reset()
        self.setParameters([bestShort, bestLong])
        done = 0
        while done == 0:
            _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))

        # If required, render the trading environment backtested
        if rendering:
            trainingEnv.render()
        
        # If required, plot the training results
        if plotTraining:
            self.plotTraining(results, bounds, step, trainingEnv.marketSymbol)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('MATF')

        # Return the trading environment backtested (training set)
        return trainingEnv


    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        """
        GOAL: Test the trading strategy on another unknown trading
              environment (called testing set) in order to evaluate
              the trading strategy performance.
        
        INPUTS: - testingEnv: Unknown trading environment (testing set).
                - trainingEnv: Known trading environment (training set).
                - rendering: Enable the trading environment rendering.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - testingEnv: Trading environment backtested.
        """

        # Execution of the trading strategy on the trading environment
        testingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = testingEnv.step(self.chooseAction(testingEnv.state))

        # If required, render the trading environment backtested
        if rendering:
            testingEnv.render()

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data,)
            analyser.displayPerformance('MATF')

        # Return the trading environment backtested (testing set)
        return testingEnv


    def plotTraining(self, results, bounds, step, marketSymbol):
        """
        GOAL: Plot both 2D and 3D graphs illustrating the results of the entire
              training phase set of simulations, depicting the performance
              associated with each combination of parameters.

        INPUTS: - results: Results of the entire set of simulations (training).
                - bounds: Bounds of the parameter search space.
                - step: Step of the parameter search space.
                - marketSymbol: Stock market trading symbol.

        OUTPUTS: /
        """

        # Generate x, y vectors and a meshgrid for the surface plot
        x = range(bounds[0], bounds[1], step)
        y = range(bounds[0], bounds[1], step)
        xx, yy = np.meshgrid(x, y, sparse=True)

        # Initialization of the 3D figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Long Window Duration')
        ax.set_ylabel('Short Window Duration')
        ax.set_zlabel('Sharpe Ratio')

        # Generate and show the 3D surface plot
        ax.plot_surface(xx, yy, results, cmap=plt.cm.get_cmap('jet'))
        ax.view_init(45, 45)
        plt.savefig(''.join(['Figures/', str(marketSymbol), '_MATFOptimization3D', '.png']))
        #plt.show()

        # Plot the same information as a 2D graph
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111,
                             ylabel='Short Window Duration',
                             xlabel='Long Window Duration')
        graph = ax.imshow(results,
                          cmap='jet',
                          extent=(bounds[0], bounds[1], bounds[1], bounds[0]))
        plt.colorbar(graph)
        plt.gca().invert_yaxis()
        plt.savefig(''.join(['Figures/', str(marketSymbol), '_MATFOptimization2D', '.png']))
        #plt.show()
    


###############################################################################
############################ Class MovingAveragesMR ###########################
###############################################################################

class MovingAveragesMR(tradingStrategy):
    """
    GOAL: Implement a Mean Reversion trading strategy based on moving averages.
        
    VARIABLES: - parameters: Trading strategy parameters, which are the windows
                             durations of the moving averages.
          
    METHODS: - __init__: Object constructor initializing the strategy parameters.
             - setParameters: Set new values for the parameters of the trading
                              strategy, which are the two windows durations.
             - processState: Process the trading environment state to obtain 
                             the required format.
             - chooseAction: Make a decision regarding the next trading
                             position (long=1 and short=0).
             - training: Train the trading strategy on a known trading
                         environment (called training set) in order to
                         tune the trading strategy parameters.
             - testing: Test the trading strategy on another unknown trading
                        environment (called testing set) in order to evaluate
                        the trading strategy performance.
    """

    def __init__(self, parameters=[5, 10]):
        """
        GOAL: Object constructor initializing the strategy parameters. 
        
        INPUTS: - parameters: Trading strategy parameters, which are the windows
                              durations of the moving averages.    
        
        OUTPUTS: /
        """

        self.parameters = parameters


    def setParameters(self, parameters):
        """
        GOAL: Set new values for the parameters of the trading strategy, 
              which are the two windows durations.
        
        INPUTS: - parameters: List of new parameters to set.      
        
        OUTPUTS: /
        """

        self.parameters = parameters

    
    def processState(self, state):
        """
        GOAL: Process the trading environment state to obtain the required format.
        
        INPUTS: - state: State of the trading environment.
        
        OUTPUTS: - state: State of the trading environment in the required format.
        """

        return state[0]


    def chooseAction(self, state):
        """
        GOAL: Make a decision regarding the next trading position
              (long=1 and short=0) based on the moving averages.
        
        INPUTS: - state: State of the trading environment.      
        
        OUTPUTS: - action: Trading position decision (long=1 and short=0).
        """

        # Processing of the trading environment state
        state = self.processState(state)

        # Computation of the two moving averages
        shortAverage = np.mean(state[-self.parameters[0]:])
        longAverage = np.mean(state[-self.parameters[1]:])

        # Comparison of the two moving averages
        if(shortAverage <= longAverage):
            # Long position
            return 1
        else:
            # Short position
            return 0
    

    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        """
        GOAL: Train the trading strategy on a known trading environment
              (called training set) in order to tune the trading strategy
              parameters, by simulating many combinations of parameters.
        
        INPUTS: - trainingEnv: Known trading environment (training set).
                - trainingParameters: Additional parameters associated
                                      with the training phase simulations.   
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the trading environment rendering.
                - plotTraining: Enable the plotting of the training results.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - trainingEnv: Trading environment associated with the best
                                trading strategy parameters backtested.
        """

        # Compute the dimension of the parameter search space
        bounds = trainingParameters[0]
        step = trainingParameters[1]
        dimension = math.ceil((bounds[1] - bounds[0])/step)

        # Initialize some variables required for the simulations
        trainingEnv.reset()
        results = np.zeros((dimension, dimension))
        bestShort = 0
        bestLong = 0
        bestPerformance = -100
        i = 0
        j = 0
        count = 1

        # If required, compute the number of simulation iterations
        if verbose:
            iterations = dimension - 1
            length = 0
            while iterations > 0:
                length += iterations
                iterations -= 1

        # Loop through all the parameters combinations included in the parameter search space
        for shorter in range(bounds[0], bounds[1], step):
            for longer in range(bounds[0], bounds[1], step):

                # Obvious restriction on the parameters
                if(shorter < longer):

                    # If required, print the progression of the training
                    if(verbose):       
                        print("".join(["Training progression: ", str(count), "/", str(length)]), end='\r', flush=True)

                    # Apply the trading strategy with the current combination of parameters
                    self.setParameters([shorter, longer])
                    done = 0
                    while done == 0:
                        _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))

                    # Retrieve the performance associated with this simulation (Sharpe Ratio)
                    performanceAnalysis = PerformanceEstimator(trainingEnv.data)
                    performance = performanceAnalysis.computeSharpeRatio()
                    results[i][j] = performance

                    # Track the best performance and parameters
                    if(performance > bestPerformance):
                        bestShort = shorter
                        bestLong = longer
                        bestPerformance = performance
                    
                    # Reset of the trading environment
                    trainingEnv.reset()
                    count += 1

                j += 1
            i += 1
            j = 0

        # Execute once again the strategy associated with the best parameters simulated
        trainingEnv.reset()
        self.setParameters([bestShort, bestLong])
        done = 0
        while done == 0:
            _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))

        # If required, render the trading environment backtested
        if rendering:
            trainingEnv.render()
        
        # If required, plot the training results
        if plotTraining:
            self.plotTraining(results, bounds, step, trainingEnv.marketSymbol)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('MAMR')

        # Return the trading environment backtested (training set)
        return trainingEnv


    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        """
        GOAL: Test the trading strategy on another unknown trading
              environment (called testing set) in order to evaluate
              the trading strategy performance.
        
        INPUTS: - testingEnv: Unknown trading environment (testing set).
                - trainingEnv: Known trading environment (training set).
                - rendering: Enable the trading environment rendering.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - testingEnv: Trading environment backtested.
        """

        # Execution of the trading strategy on the trading environment
        testingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = testingEnv.step(self.chooseAction(testingEnv.state))

        # If required, render the trading environment backtested
        if rendering:
            testingEnv.render()

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('MAMR')

        # Return the trading environment backtested (testing set)
        return testingEnv


    def plotTraining(self, results, bounds, step, marketSymbol):
        """
        GOAL: Plot both 2D and 3D graphs illustrating the results of the entire
              training phase set of simulations, depicting the performance
              associated with each combination of parameters.

        INPUTS: - results: Results of the entire set of simulations (training).
                - bounds: Bounds of the parameter search space.
                - step: Step of the parameter search space.
                - marketSymbol: Stock market trading symbol.

        OUTPUTS: /
        """

        # Generate x, y vectors and a meshgrid for the surface plot
        x = range(bounds[0], bounds[1], step)
        y = range(bounds[0], bounds[1], step)
        xx, yy = np.meshgrid(x, y, sparse=True)

        # Initialization of the 3D figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Long Window Duration')
        ax.set_ylabel('Short Window Duration')
        ax.set_zlabel('Sharpe Ratio')

        # Generate and show the surface 3D surface plot
        ax.plot_surface(xx, yy, results, cmap=plt.cm.get_cmap('jet'))
        ax.view_init(45, 45)
        plt.savefig(''.join(['Figures/', str(marketSymbol), '_MAMROptimization3D', '.png']))
        #plt.show()

        # Plot the same information as a 2D graph
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111,
                             ylabel='Short Window Duration',
                             xlabel='Long Window Duration')
        graph = ax.imshow(results,
                          cmap='jet',
                          extent=(bounds[0], bounds[1], bounds[1], bounds[0]))
        plt.colorbar(graph)
        plt.gca().invert_yaxis()
        plt.savefig(''.join(['Figures/', str(marketSymbol), '_MAMROptimization2D', '.png']))
        #plt.show()
        