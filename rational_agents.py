# Isaac Joffe (2024)


# basic built-in libraries required
import numpy as np
# code developed for basic utilities
from agents import *


# agent that randomly decides to cooperate or defect each round
class RandomAgent(Agent):
    def play(self):
        return Choice(np.random.randint(2))


# agent that always cooperates
class NiceAgent(Agent):
    def play(self):
        return Choice.COOPERATE


# agent that always defects
class NastyAgent(Agent):
    def play(self):
        return Choice.DEFECT


# agent that follows lead of other agent
class TitForTatAgent(Agent):
    def play(self):
        history = self.get_history()
        if history:
            # if not in first round, copy whatever opponent did last round (other position)
            return history[-1].get_choices()[1 - self.get_position()]
        # start by cooperating
        return Choice.COOPERATE


# agent that follows lead of other agent, but nicer (from https://ncase.me/trust/)
class TitForTwoTatsAgent(Agent):
    def play(self):
        history = self.get_history()
        if (history) and (len(history) > 1):
            # if not in first two rounds, defect only if opponent has defected twice in a row
            if (history[-2].get_choices()[1 - self.get_position()] == Choice.DEFECT) and (history[-1].get_choices()[1 - self.get_position()] == Choice.DEFECT):
                return Choice.DEFECT
        # start by cooperating
        return Choice.COOPERATE


# agent that cooperates until the other defects, then always defects (from https://ncase.me/trust/)
class GrudgerAgent(Agent):
    def play(self):
        history = self.get_history()
        if history:
            for round in history:
                # defect if opponent has ever defected before
                if (round.get_choices()[1 - self.get_position()] == Choice.DEFECT):
                    return Choice.DEFECT
        # start by cooperating
        return Choice.COOPERATE


# complex rule-based agent (from https://ncase.me/trust/)
class DetectiveAgent(Agent):
    def play(self):
        history = self.get_history()
        # send out initial feeler to gauge other agent
        if len(history) == 0:
            return Choice.COOPERATE
        elif len(history) == 1:
            return Choice.DEFECT
        elif len(history) == 2:
            return Choice.COOPERATE
        elif len(history) == 3:
            return Choice.COOPERATE
        # if they ever cheated, then play tit-for-tat
        else:
            for round in history:
                if (round.get_choices()[1 - self.get_position()] == Choice.DEFECT):
                    return history[-1].get_choices()[1 - self.get_position()]
            return Choice.DEFECT
