# Isaac Joffe (2024)


# basic built-in libraries required
import numpy as np
# code developed for basic utilities
from agents import *


# one flip occurs roughly every this amount of rounds
noise_param = 40
cognitive_param = 10


# agent that acts based on Picard model of emotion
class PicardAgent(Agent):
    # pass in personality parameters of agent
    def __init__(self, name, params):
        super().__init__(name)
        self.__a = params["a"]
        self.__b = params["b"]
        # assume starting from neutral mood
        self.__c = 0
        # assume starting with no expectations at all
        self.__d = 0
        self.__e = params["e"]
        return

    def play(self):
        # Adapted version of model applied here:
        #   y = 2a / 1 + e^(−b(x + c)) - a - d >= −e
        # y is output: larger more positive
        # x is input: larger more positive
        # a is arousal: larger more emotional
        # b is temperament: larger swings more wildly
        # c is mood: larger more positive
        # d is cognitive expectation: larger more positive
        # e is decision threshold: larger more positive

        history = self.get_history()
        # input is last decision, or 0 if in first round
        x = history[-1].get_choices()[1 - self.get_position()].get_value() if history else 0
        # update cognitive expectation based on exponential deterioration and include new decision
        self.__d /= 2
        self.__d += (history[-1].get_choices()[1 - self.get_position()].get_value() if len(history) >= 2 else 0) / cognitive_param
        # directly apply model formula
        y = 2 * self.__a / (1 + np.exp(-self.__b * (x + self.__c))) - self.__a - self.__d
        choice = (Choice.COOPERATE if y >= -self.__e else Choice.DEFECT)
        # print(f"Parameters:\n  x: {x}\n  a: {self.__a}\n  b: {self.__b}\n  c: {self.__c}\n  d: {self.__d}\n  e: {self.__e}\nResult:\n  y: {y}\n  choice: {choice}")
        # new mood is current emotion activation
        self.__c = y
        # sometimes random noise changes decision
        if np.random.randint(noise_param) == 0:
            choice = (Choice.COOPERATE if choice == Choice.DEFECT else Choice.DEFECT)
        return choice
