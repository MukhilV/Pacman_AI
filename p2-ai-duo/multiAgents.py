# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPacmanPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        currentGhostStates = currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # The successor state is the goal state, return big reward
        if successorGameState.isWin():
            return 1e6
        
        # Variable initializations
        score, shortestDistToNextFood = 0, 1e6

        newFoodPositions = newFood.asList()
        numberOfFoods = len(newFoodPositions)

        # Computing shortest distance to the next food (or) finding the next closest food
        for foodPosition in newFoodPositions:
          distance = manhattanDistance(foodPosition,newPacmanPos) + numberOfFoods*100
          if distance < shortestDistToNextFood: shortestDistToNextFood = distance

        # If there are no foods, then it is the goal state, hence don't change the score.
        if numberOfFoods == 0 : shortestDistToNextFood = 0

        # Subtract the value of shortest distance to next food, if there are more foods to be eaten
        score -= shortestDistToNextFood

        # Power Pellets are good states, hence increase the score
        numberofPowerPellets = len(successorGameState.getCapsules())
        if newPacmanPos in currentGameState.getCapsules(): score += 150 * numberofPowerPellets

        currentDistanceToGhost = [manhattanDistance(newPacmanPos, ghostState.getPosition()) for ghostState in currentGhostStates]
        newDistanceToGhosts = [manhattanDistance(newPacmanPos, ghostState.getPosition()) for ghostState in newGhostStates]  

        
        sumOfScaredTimes = sum(newScaredTimes)
        # It is better to be near the ghost when they are scared
        if sumOfScaredTimes > 0 :
            if min(currentDistanceToGhost) < min(newDistanceToGhosts): score += 200
            else: score -=100

        # It is better to be away from the ghost when they aren't scared
        else:
            if min(currentDistanceToGhost) < min(newDistanceToGhosts): score -= 100
            else: score += 200

        # If the next Pacman Position collides with the any positions of ghost, return a big negative reward
        for i in range(len(newGhostStates)):
          ghostPos = successorGameState.getGhostPosition(i+1)
          if manhattanDistance(newPacmanPos,ghostPos)<=1 :
            score -= 1e6

        return score 
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"


        def maxLevel(state, agentIndex, depth):
            # if it is win/lose state or at maximum_depth, return existing evaluation function 
            if  state.isWin() or state.isLose() or depth == self.depth :
                return self.evaluationFunction(state)
        
            agentIndex = PACMAN_AGENT_INDEX
            legalActions = state.getLegalActions(agentIndex)

            # for every action in maxlevel, find its minlevel value at next depth and return the max of all values
            return max([minLevel(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1) 
                                 for action in legalActions])
        
        def minLevel(state, agentIndex, depth):
            # if it is win/lose state, return existing evaluation function 
            if state.isWin() or state.isLose(): return self.evaluationFunction(state)
            
            legalGhostActions = state.getLegalActions(agentIndex)
            if agentIndex == numberOfAgents - 1:
                minimumValue = float('inf')
                for action in legalGhostActions: 
                    currMin = maxLevel(state.generateSuccessor(agentIndex, action), agentIndex,  depth)
                    minimumValue = currMin if(currMin < minimumValue) else minimumValue
                return minimumValue
                # return min([maxLevel(state.generateSuccessor(agentIndex, action), 
                #                              agentIndex,  depth) for action in legalGhostActions])
            else:
                # maximumValue = float('-inf')
                # for action in legalGhostActions:
                #     currMax = minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
                #     maximumValue = currMax if(currMax > maximumValue) else maximumValue
                # return maximumValue
                return min([minLevel(state.generateSuccessor(agentIndex, action),
                                            agentIndex + 1, depth) for action in legalGhostActions])

        PACMAN_AGENT_INDEX = 0
        numberOfAgents = gameState.getNumAgents()
        legalPacmanActions = gameState.getLegalActions(0)
        actionValuePair = []

        # For every action, append the action and its minimax value into a list
        for action in legalPacmanActions:
            actionValuePair.append((minLevel(gameState.generateSuccessor(0, action), 1, 1), action))

        # return the last element from the sorted list, which is max
        return sorted(actionValuePair)[-1][1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxLevel(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose()  or depth == self.depth: return self.evaluationFunction(state)
            agentIndex = PACMAN_AGENT_INDEX
            legalPacmanActions = state.getLegalActions(agentIndex)
 
            maximumValue = -1e6
            alpha1 = alpha

            for action in legalPacmanActions:
                maximumValue = max(maximumValue, minLevel(state.generateSuccessor(agentIndex, action), \
                agentIndex + 1, depth + 1, alpha1, beta) )
                if maximumValue > beta: return maximumValue
                alpha1 = max(alpha1, maximumValue)
            return maximumValue
        
        def minLevel(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose(): return self.evaluationFunction(state)
            agentCount = gameState.getNumAgents()
            legalGhostActions = state.getLegalActions(agentIndex)

            minimumValue = 1e6
            beta1 = beta
            
            if agentIndex == agentCount - 1:
                for action in legalGhostActions:
                    minimumValue =  min(minimumValue, maxLevel(state.generateSuccessor(agentIndex, action), 
                                                               agentIndex,  depth, alpha, beta1))
                    if minimumValue < alpha: return minimumValue
                    beta1 = min(beta1, minimumValue)

            else:
                for action in legalGhostActions:
                    minimumValue =  min(minimumValue,minLevel(state.generateSuccessor(agentIndex, action), 
                                                              agentIndex + 1, depth, alpha, beta1))
                    if minimumValue < alpha: return minimumValue
                    beta1 = min(beta1, minimumValue)

            return minimumValue



        PACMAN_AGENT_INDEX = 0
        PacmanActions = gameState.getLegalActions(0)
        alpha = -1e6
        beta = 1e6
        actionValuePairs = []
        for action in PacmanActions:
            successor = gameState.generateSuccessor(0, action)
            currValue = minLevel(successor, 1, 1, alpha, beta)
            actionValuePairs.append((currValue,action))

            if currValue > beta: return action
            alpha = max(currValue, alpha)

        return sorted(actionValuePairs)[-1][1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expValue(state, agentIndex, depth):
            if state.isWin() or state.isLose(): return self.evaluationFunction(state)

            numberOfGhosts = gameState.getNumAgents()
            legalGhostActions = state.getLegalActions(agentIndex)

            expectedValue = 0
            PofA = 1.0 / len(legalGhostActions) 
            expectedValues = []
            for action in legalGhostActions:
                if agentIndex == numberOfGhosts - 1:
                    expectedValues.append(maxValue(state.generateSuccessor(agentIndex, action),
                                                agentIndex,  depth) * PofA)
                else:
                    expectedValues.append(expValue(state.generateSuccessor(agentIndex, action),
                                               agentIndex + 1, depth) * PofA)
                expectedValues.append(expectedValue)
            return sum(expectedValues)


        def maxValue(state, agentIndex, depth):
            if state.isWin() or state.isLose()  or depth == self.depth: return self.evaluationFunction(state)
       
            agentIndex = PACMAN_AGENT_INDEX
            legalPacmanActions = state.getLegalActions(agentIndex)

            return max([expValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1) 
                        for action in legalPacmanActions])

        PACMAN_AGENT_INDEX = 0
        actions = gameState.getLegalActions(0)
        actionValueDict = {}
        for action in actions:
            actionValueDict[action] = expValue(gameState.generateSuccessor(0, action), 1, 1)

        return max(actionValueDict, key=actionValueDict.get)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
