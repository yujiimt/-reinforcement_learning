# -*- coding: utf-8 -*-

class Planner():

    def __init__(self, env):
        self.env = env
        self.log = []
    
    def initialize(self):
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("planner have to implements plan method")

    def transition_at(self, state, action):
        transition_probs = self.env.transition_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward
    
    def dict_to_grid(self, state_reward_dict):
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]
        return grid

class ValueIterationPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        actions = self.env.actions
        V = {}
        for s in self.env.states:
            V[s] = 0
        
        while True:
            # deltaが更新幅
            delta = 0
            self.log.append(self.dict_to_grid(V))
            for s in V:
                if not self.env.can_action_at(s):
                    continue
                expected_rewards = []
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transition_at(s, a):
                        r += prob*(reward + gamma * V[next_state])
                    expected_rewards.append(r)
                # 各状態の各行動に対して価値の計算を行い、最大値で更新をおこなっている
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward
            
            # 価値の更新幅delta が threshold を下回るまで更新を続ける
            if delta < threshold:
                break
        V_grid = self.dict_to_grid(V)
        return V_grid

class PolicyIterationPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)
        self.policy = {}

    def initialize(self):
        super().initialize()
        self.policy = {}
        actions = self.env.actions
        states = self.env.states
        for s in states:
            self.policy[s] = {}
            for a in actions:
                self.policy[s][a] = 1 / len(actions)
    #戦略による価値の計算
    def estimated_by_policy(self, gamma, threshold):
        V = {}
        for s in self.env.states:
            V[s] = 0
        
        while True:
            delta = 0
            for s in V:
                expected_rewards = []
                for a in self.policy[s]:
                    action_prob = self.policy[s][a]
                    r = 0
                    for prob, next_state, reward in self.transition_at(s, a):
                        r += action_prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward
            if delta < threshold:
                break 
        return V