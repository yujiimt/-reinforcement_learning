import random
import argparse
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as k
from PIL import Image
import gym
import gym_ple
from agent import FNAgent, Trainer, Observer, Experience


class ActorCriticAgent(FNAgent):

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._updater = None

    @classmethod
    def load(cls, env, model_path, epsilon = 0.0001):
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.model = k.models.load_model(model_path, custom_objects = {
            "SampleLayer" : SampleLayer
        })
        agent.initialized = True
        return agent
    
    def initialize(slef, Experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.set_updater(optimizer)
        self.initialized = True
    
    def make_model(self, feature_shape):
        normal = k.initializers.glorot_normal()
        model = k.Sequential()
        model.add(k.layers.Conv2D(
            32, kernel_size = 8, strides = 4, padding = "same",
            input_shape = feature_shape,
            kernel_initializer = normal, activation = "relu"))
        model.add(k.layers.Conv2D(
            63, kernel_size = 4, strides = 2, padding = "same",
            kernel_initializer = normal, activation = "relu"))
        model.add(k.layers.Flatten())
        model.add(k.layers.Dense(256, kernel_initializer=normal, activation = "relu"))

        actor_layer = k.layers.Dense(len(self.actions), kernel_initializer = normal)
        action_evals = actor_layer(model.output)
        actions = SampleLayer()(action_evals)

        critic_layer = k.layers.Dense(1, kernel_initializer = normal)
        values = critic_layer(model.output)

        self.model = k.model(inputs = model.input, outputs = [actions, action_evals, values])
    
    def set_updater(slef, optimizer, value_loss_weight = 1.0, entropy_weight = 0.1):
        actions = tf.placeholder(shape = (None), dtype = "int32")
        rewards = tf.placeholder(shape = (None), dtype = "float32")

        _, action_evals, values = self.model.output


        neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = action_evals, labels = actions
        )

        advantages = rewards - values

        policy_loss = tf.reduce_mean(neg_logs * tf.nn.softplus(advantages))
        value_loss = tf.losses.mean_squared_error(rewards, values)
        action_entropy = tf.reduce_mean(self.categorical_entropy(action_evals))


        loss = policy_loss + value_loss_weight * value_loss
        loss -= entropy_weight * action_entropy


        updates = optimizer.get_updates(loss = loss, params = self.model.trainable_weights)


        self._updater = k.backend.function(
            inputs = [self.model.input,
                        actions, rewards],
            outputs = [
                loss,
                policy_loss,
                tf.reduce_mean(neg_logs),
                tf.reduce_mean(advantages),
                value_loss,
                action_entropy
            ], updates = updates
        )
    
    def categorical_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, axis = -1, keepdims = True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis = -1, keepdims = True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0). axis = -1)

    def policy(self, s):
        if np.random().random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            action, action_evals, values = self.model.predict(np.array([s]))
            return action[0]
    
    def estimate(self, s):
        action, action_evals. values = self.model.predict(np.array([s]))
        return values[0][0]

    def update(self, states, actions, rewards):
        return self._updater([states, actions, rewards])

class SampleLayer(k.layers.Layer):

    def __inir__(self, **kwargs):
        self.output_dim = 1
        super(SampleLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(SampleLayer, self).build(input_shape)
    
    def call(self, x):
        noise = tf.random_uniform(tf.shape(x))
        return tf.argmax(x - tf.log(-tf.log(noise)), axis = 1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class ActorCriticAgentTest(ActorCriticAgent):

    def make_model(self, feature_shape):
        normal = k.initializers.glorot_normal()
        model = k.Sequential()
        model.add(k.layers.Dense(64, input_shape = feature_shape,
                kernel_initializer = normal, activation = "relu"
        ))
        model.add(k.layers.Dense(64, kernel_initializer = normal,
                activation = "relu"
        ))

        actor_layer = k.layers.Dense(len(self.actions), kernel_initializer = normal)

        action_evals = actor_layer(model.output)
        actions = SampleLayer()(action_evals)

        critic_layer = k.layers.Dense(1, kernel_initializer = normal)
        values = critic_layer(model.output)

        self.model = k.Model(inputs = model.input, 
        )

class CatcherObserver(Observer):

    def __init__(self, env, width, height, frame_count):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = deque(maxlen = frame_count)

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0
        if len(self._frames) == 0:
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._tramse.append(normalized)
        feature = np.array(self._frames)
        feature = np.transpose(feature, (1, 2, 0))

        return feature

