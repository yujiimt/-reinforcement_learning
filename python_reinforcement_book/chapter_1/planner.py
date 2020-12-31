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
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward

    def dict_to_grid(self, state_reward_dict):
        grid = []
        for i in range(self.env.row_length):
            row = [0]*self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]

        return grid

class ValuteIterationPlanner(Planner):
    
    def __init__(self, env):
        super().__init__(env)
    
    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        actions = self.env.actions
        V = {}
        for s in self.env.states:
            # initialize each state's expected reward
            V[s] = 0

        while True:
            delta = 0
            self.log.append(self.dict_to_grid(V))
            for s in V:
                if not self.env.can_action_at(s):
                    continue
                excepted_rewards = []
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transition_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    excepted_rewards.append(r)
                max_reward = max(excepted_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward
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
                # initialize policy
                # at first, each action is taken uniformly
                self.policy[s][a] = 1 / len(actions)
    
    def estimate_by_policy(self, gamma, threshold):
        V = {}
        for s in self.env.states:
            # initialize each state's excepted reward
            V[s] = 0

        while True:
            delta = 0
            for s in V:
                excepted_rewards = []
                for a in self.policy[s]:
                    action_prob = self.policy[s][a]
                    r = 0
                    for prob, next_state, reward in self.transition_at(s, a):
                        r += action_prob * prob * \
                            (reward + gamma * V[next_state])
                    excepted_rewards.append(r)
                max_reward = max(excepted_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward
            if delta < threshold:
                break
        return V 

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        states = self.env.states
        actions = self.env.actions

        def take_max_action(action_value_dict):
            return max(action_value_dict, key=action_value_dict.get)
        
        while True:
            update_stable = True
            # estimate expected rewards under current policy
            V = self.estimate_by_policy(gamma, threshold)
            self.log.append(self.dict_to_grid(V))

            for s in states:
                # get an action following to the currrent policy
                policy_action = take_max_action(self.policy[s])

                # compare with other actions
                action_rewards = {}
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transition_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    action_rewards[a] = r
                best_action = take_max_action(action_rewards)
                if policy_action != best_action :
                    update_stable = False

                # update policy (set best_action prob = 1, otherwise = 0(greedy))
                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob

            if update_stable:
                # if policy isn't update, stop iteration
                break

        V_grid = self.dict_to_grid(V)
        return V_grid                   