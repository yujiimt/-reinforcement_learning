# -*- coding: utf-8 -*-


import numpy as np
from enum import Enum

class State():
    """
    state はセルの位置（row, column）で、ACtionは上下左右の行動を表現されている
    """
    def __init__(self, row=1, columns=1):
        self.row = row
        self.column = columns

    def __repr__(self):
        return "<State : [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))
    
    def __eq__(self, other):
        return self.row == other.row and self.column == other.column

class Action(Enum):
     UP = 1
     DOWN = -1
     LEFT = 2
     RIGHT = -2


class Enviroment():
    """
    Enviroment は迷路の定義（grid）を受け取り、
    迷路内のセルを環境における状態とする
    """

    def __init__(self, grid, move_prob = 0.8):
        self.grid = grid
        self.agent_state = State()
        self.default_reward = -0.04

        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, 
                    Action.LEFT, Action.RIGHT]
    
    @property 
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                if self.grid[row][column] != 9 :
                    states.append(State(row, column))
        return states

    # 遷移関数↓
    def transit_func(self, state, action):
        transistion_probs = {}

        if not self.can_action_at(state):
            return transistion_probs

        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            if next_state not in transistion_probs:
                transistion_probs[next_state] = prob
            else:
                transistion_probs[next_state] += prob
        return transistion_probs

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here")
        next_state = state.clone()

        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state
        
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state
   #遷移確率は、選択した行動には self.move_prob, 
   #逆方向以外の行動には残りの確率を等分した確率( 1 - self.move_prob) / 2  が割り当てられる
   #遷移先は選択された方向に移動したセルになりますが、迷路の範囲外に出る場合は元のセルに戻されます
   #こうした移動処理は　_moveで実装される。報酬関数を見てましょう 

   # 報酬関数　↓
    def reward_func(self, state):
       reward = self.default_reward
       done = False
       attribute = self.grid[state.row][state.column]
       if attribute == 1:
           reward = 1
           done = True
       elif attribute == -1:
            reward = -1
            done = True

       return reward, done
    
    def reset(self):
        self.agent_state = State(self.row_length - 1 , 0)
        return self.agent_state

    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state
        return next_state, reward, done
    
    def transit(self, state, action):
        transistion_probs = self.transit_func(state, action)
        if len(transistion_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transistion_probs:
            next_states.append(s)
            probs.append(transistion_probs[s])

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done