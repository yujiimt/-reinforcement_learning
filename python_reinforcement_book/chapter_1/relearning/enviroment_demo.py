import random
import ray
from enviroment import Enviroment

ray.init()

class Agent():
    
    def __init__(self, env):
        self.actions = env.actions
    
    def policy(self, state):
        return random.choice(self.actions)

@ray.remote
def main():
    grid = [
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 9, 0, -1, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 1, 1, 1, 1, -1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, -1],
        [0, 0, 0, 0, 1, 1, 1, 1, -1]
    ]

    env = Enviroment(grid)
    agent = Agent(env)

    for i in range(100000000):
        state = env.reset()
        total_reward = 0
        done = False

    while not done:
        action = agent.policy(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
    
    print("Episode {}: Agent gets {} reaward".format(i, total_reward))


if __name__ == "__main__":
    main().remote()

