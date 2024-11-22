# Isaac Joffe (2024)


# basic built-in libraries required
import numpy as np
import copy
# other code developed for project
from agents import *
from rational_agents import *
from emotional_agents import *


# represents a single round of the PD
class PrisonersDilemma():
    # store information about the agents, reward structure, and results
    def __init__(self, iteration, reward_matrix, agent_a, agent_b):
        self.__iteration = iteration
        self.__reward_matrix = reward_matrix
        self.__agent_a = agent_a
        self.__agent_b = agent_b
        self.__choices = None
        self.__scores = None
        return

    # return the choices made by each agent this round
    def get_choices(self):
        return self.__choices

    # return the scores achieved by each agent this round
    def get_scores(self):
        return self.__scores

    # determine the scores acheieved by each agent this round
    def compute_rewards(self, choices):
        scores = (None, None)
        if (choices[0] == Choice.COOPERATE) and (choices[1] == Choice.COOPERATE):
            # both get second-maximum reward
            scores = (self.__reward_matrix[2], self.__reward_matrix[2])
        elif (choices[0] == Choice.COOPERATE) and (choices[1] == Choice.DEFECT):
            # cooperator gets minimal reward, defector gets maximal reward
            scores = (self.__reward_matrix[0], self.__reward_matrix[3])
        elif (choices[0] == Choice.DEFECT) and (choices[1] == Choice.COOPERATE):
            # defector gets maximal reward, cooperator gets minimal reward
            scores = (self.__reward_matrix[3], self.__reward_matrix[0])
        elif (choices[0] == Choice.DEFECT) and (choices[1] == Choice.DEFECT):
            # both get second-minimum reward
            scores = (self.__reward_matrix[1], self.__reward_matrix[1])
        return scores

    # play a round of the PD game
    def play(self):
        # each agent makes its decision, gets corresponding scores
        self.__choices = self.__agent_a.play(), self.__agent_b.play()
        self.__scores = self.compute_rewards(self.__choices)
        # save results into memroy to inform future rounds
        self.__agent_a.update_history(self), self.__agent_b.update_history(self)
        return self


# represents many rounds of the PD
class IteratedPrisonersDilemma():
    # set up game with length, structure, and players
    def __init__(self, n_iterations, reward_matrix, agent_a, agent_b):
        self.__n_iterations = n_iterations
        self.__reward_matrix = reward_matrix
        self.__agent_a = agent_a
        self.__agent_b = agent_b
        self.__history = [0] * self.__n_iterations
        self.__scores = (0, 0)
        return

    # return the scores achieved by each agent across all rounds
    def get_scores(self):
        return self.__scores

    # return the log of all rounds the agent has played in the IPD
    def get_history(self):
        return self.__history

    # display round-by-round results of this IPD game
    def print_results(self):
        # history = self.get_history()
        # for i in range(self.__n_iterations):
        #     print(f"Round {i}:  {self.__agent_a} {history[i].get_choices()[0]} for {history[i].get_scores()[0]} points, {self.__agent_b} {history[i].get_choices()[1]} for {history[i].get_scores()[1]} points")
        # print(f"{self.__agent_a} Score: {self.get_scores()[0]} ({(self.get_scores()[0] / len(history)):.2f} per round)")
        # print(f"{self.__agent_b} Score: {self.get_scores()[1]} ({(self.get_scores()[1] / len(history)):.2f} per round)")
        return

    # play the PD game for the desired number of rounds
    def play(self):
        for i in range(self.__n_iterations):
            # play the current iteration of the PD, previous rounds knowledge stored in agents
            round = PrisonersDilemma(
                i,
                self.__reward_matrix,
                self.__agent_a,
                self.__agent_b,
            )
            self.__history[i] = round.play()
            # log total scores for each agent
            self.__scores = (self.__scores[0] + self.__history[i].get_scores()[0], self.__scores[1] + self.__history[i].get_scores()[1])
        return self


# represents a tournament of many agents playing the IPD
class Tournament():
    # set up tournament with IPD parameters and agents to test
    def __init__(self, n_iterations, reward_matrix, agents):
        self.__n_iterations = n_iterations
        self.__reward_matrix = reward_matrix
        self.__agents = agents
        self.__matchups = []
        self.__scores = [0] * len(self.__agents)
        return

    # return the total scores achieved by each agent across all matchups
    def get_scores(self):
        return self.__scores

    # display overall results of the tournament
    def print_results(self):
        # print()
        # print("----------------------------------------------------")
        # print("---------------- Tournament Results ----------------")
        # print("----------------------------------------------------")
        # scores = self.get_scores()
        # for i in range(len(self.__agents)):
        #     print(f"Agent {self.__agents[i]}: {scores[i]} points ({(scores[i] / (len(self.__agents) + 1)):.2f} per game, {(scores[i] / (len(self.__agents) + 1) / self.__n_iterations):.2f} per round)")
        # print("----------------------------------------------------")
        # print()
        return

    # play a round robin of the IPD
    def play(self):
        # each agent plays all others once and itself
        k = 0
        for i in range(len(self.__agents)):
            for j in range(i, len(self.__agents)):
                # print information to track progress of tournament
                # print("----------------------------------------------------")
                # print(f"----- Matchup {k}:  {self.__agents[i]} vs. {self.__agents[j]} -----")
                # print("----------------------------------------------------")
                k += 1

                # create fresh version of agents to play the game
                agent_a = copy.deepcopy(self.__agents[i])
                agent_b = copy.deepcopy(self.__agents[j])
                agent_a.reset(0)
                agent_b.reset(1)

                # run the IPD between these agents
                matchup = IteratedPrisonersDilemma(
                    self.__n_iterations,
                    self.__reward_matrix,
                    agent_a,
                    agent_b,
                )
                self.__matchups.append(matchup.play())

                # track results of tournament
                self.__scores[i] += self.__matchups[-1].get_scores()[0]
                self.__scores[j] += self.__matchups[-1].get_scores()[1]

                # display results of each matchup
                self.__matchups[-1].print_results()

        # display summary of entire tournament results
        # print("----------------------------------------------------")
        self.print_results()

        # return scores for subsequent data analysis
        return self.get_scores()


# represents a game between a certain agent and the user
class UserGame():
    # set up tournament with IPD parameters and agent to test
    def __init__(self, n_iterations, reward_matrix, agent):
        self.__n_iterations = n_iterations
        self.__reward_matrix = reward_matrix
        self.__agent = agent
        self.__matchup = None
        return

    # display overall results of the tournament
    def print_results(self):
        print()
        print("----------------------------------------------------")
        print("---------------- Tournament Results ----------------")
        print("----------------------------------------------------")
        scores = self.__matchup.get_scores()
        print(f"You: {scores[0]} points ({(scores[0] / self.__n_iterations):.2f} per round)")
        print(f"Agent {self.__agent}: {scores[1]} points ({(scores[1] / self.__n_iterations):.2f} per round)")
        print("----------------------------------------------------")
        print()
        return

    # play a single game of the IPD
    def play(self):
        # create fresh version of agents to play the game
        agent_a = UserAgent("User")
        agent_b = self.__agent
        agent_a.reset(0)
        agent_b.reset(1)

        # run the IPD between the user and the agent
        self.__matchup = IteratedPrisonersDilemma(
            self.__n_iterations,
            self.__reward_matrix,
            agent_a,
            agent_b,
        )
        self.__matchup.play()
        print(f"Round {self.__n_iterations}: You {self.__matchup.get_history()[-1].get_choices()[0]}, they {self.__matchup.get_history()[-1].get_choices()[1]}")

        # display results of the overall game
        self.print_results()
        return


# play a game against the emotiona agents
def main():
    n_iterations = 10
    reward_matrix = [0, 1, 3, 5]
    UserGame(
        n_iterations,
        reward_matrix,
        PicardAgent("Emotional", {"a": 1, "b": 1, "e": 0})
    ).play()
    return


# run main function if called as program
if __name__ == "__main__":
    main()
