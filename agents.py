# Isaac Joffe (2024)


# basic built-in libraries required
from enum import Enum


# agents can either cooperate or defect in the PD
class Choice(Enum):
    # assign arbitrary integers to represent the choice
    COOPERATE = 0
    DEFECT = 1

    # get integer version of decision
    def get_value(self):
        return -2 * self.value + 1
            
    # ensure printed version is properly formatted
    def __str__(self):
        if self.value == 0:
            return "COOPERATE"
        if self.value == 1:
            return "DEFECT"


# abstract class that defines generic behavior of IPD-playing agent
class Agent():
    # all agents must keep track of past rounds and their cumulative score
    def __init__(self, name):
        self.__name = name
        self.__score = 0
        self.__position = None
        self.__history = []
        return
    
    # ensure printed version is properly formatted
    def __str__(self):
        return self.__name
    
    # return the total cumulative score for the agent in the IPD
    def get_score(self):
        return self.__score

    # return whether this agents is the first or second one
    def get_position(self):
        return self.__position

    # return the log of all rounds the agent has played in the IPD
    def get_history(self):
        return self.__history

    # set up new instantiation of agent, store where it is within the game
    def reset(self, position):
        self.__score = 0
        self.__position = position
        self.__history = []
        return

    # store results of each round to inform future decisions
    def update_history(self, round):
        # need to index prisoner's dilemma result with this agent's position
        self.__score += round.get_scores()[self.get_position()]
        self.__history.append(round)
        return

    # abstract method to play a round of the PD in the IPD, decision must be implemented by subclass
    def play(self):
        raise NotImplementedError("General agent cannot play the PD")


# agent that is controlled by user
class UserAgent(Agent):
    def play(self):
        history = self.get_history()
        # inform user of what happen in last round
        if history:
            print(f"Round {len(history)}: You {history[-1].get_choices()[self.get_position()]}, they {history[-1].get_choices()[1 - self.get_position()]}")
        # act based on whatever user provides as input
        return Choice(int(input(f"Round {len(history) + 1}: Cooperate (0) or Defect (1)?: ")))
