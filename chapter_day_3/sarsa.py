from collections import defaultdict
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value

class SARSAAgent(ELAgent):

    def __init__(sefl, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count = 1000, gamme = 0.9,
    learning_rate = 0.1. render = False, report_interval = 50):
        self.init

