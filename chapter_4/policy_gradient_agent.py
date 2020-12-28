import os
import argeparse
import random
from collections import deque
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow.python import keras as k
import gym
from agent import FNAgent, Trainer, Observer, Experience


class PolicyGradientAgent(FNAgent):


    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self.estimate_probs = True
        self.scarler = None
        self._updater = None
    
    def save(self, model_path):
        super().save(model_path)
        joblib.dump(self.scaler, self.scaler_path(model_path))
    
    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        agent = super().load(env, model_path, epsilon)
        agent.scaler = joblib.load(agent.scaler_path(model_path))
        return agent

    def scaler_path(self, model_path):
        fname, _ = os.path.splitext(model_path)
        fname += "_scaler.pkl"
        return fname
    
    def initialize(self, experience, optimizer):
        self.scaler = StandardScaler()
        states = np.vstack([e.s for e in experiences])
        self.scaler.fit(states)

        feature_size = states.shape[1]
        self.model = k.models.Sequential([
            k.layers.Dense(10, activation = "relu", input_shape = (feature_size,)),
            k.layers.Dense(10, activation = "relu"),
            k.layers.Dense(len(self.actions), activation = "softmax")
        ])
        self.set_update(optimizer)
        self.initialized = True
        print("Done initialization . From now, begin training !")

    def set_update(self, optimizer):
        actions = tf.placeholder(shape = (None), dtype = "int32")
        rewards = tf.placeholder(shape = (None), dtype = "float32")
        one_hot_actions = tf.one_hot(actions, len(self.actions), axis=1)
        action_probs = self.model.output
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs, axis = 1)
        
        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)

        loss = -tf.log(clipped) * rewards
        loss = tf.reduce_mean(loss)

        updates = optimizer.get_updates(loss=loss, params = self.model.trainable_weights)
        
        self._updater = k.backend.function(
            inputs = [self.model.input,
                      actions, rewards],
            outputs = [loss],
            updates = updates)
    
    def estimate(self, s):
        normalized = self.scaler.transform(s)
        action_probs = self.model.predict(normalized)[0]
        return action_probs

    def update(self, states, actions, rewards):
        
