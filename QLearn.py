import pdb;
import scipy.misc as scimisc

from tkinter import *
from PIL import Image
from PIL import ImageTk


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from PIL import Image

import MalmoPython
import os
import sys
import time
import random
import json
import numpy as np
import time
from IPython.display import clear_output,display
import logging
import math

import gym
import gym_minecraft
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule

def preprocess(rgb_array,scale = 1/12):
    frame_shape = rgb_array.shape
    
    frame = np.array(rgb_array)
    gray_frame = np.dot(frame[...,:3],[0.299,0.587,0.114]).reshape((frame_shape[0],frame_shape[1]))
    smaller = scimisc.imresize(gray_frame,scale,mode='L').astype('float64')
    smaller = np.expand_dims(smaller,2) # convert to a 3D array of shape (height,width,grayscale)
    return smaller

def render(obs,root,canvas):
    obs = np.squeeze(obs,2)
    image = Image.fromarray(obs.astype('int8'),mode='L')
    photo = ImageTk.PhotoImage(image)
    root.one = photo
    canvas.delete("all")
    canvas.create_image(frame_height,frame_width, image=photo)
    root.update()
    
    
def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out
qmodel = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
)


if __name__ == '__main__':
    with U.make_session(1):
        # Create the environment
        env = gym.make("MinecraftBasic-v0")
        env.init(videoResolution=[420,300],allowContinuousMovement=["move", "turn", "strafe"])
        env.load_mission_file("./CliffWalking.xml")
        scale = 1/12 # scale image down by 1/12
        newshape = (env.video_height*scale,env.video_width*scale,1) # dimension of 1 for grayscale
        newshape = tuple(map(int,newshape))

        # the pre processor will adjust the observation space therefore we will edit the property of the environment to take the pre processor into accoutn
        env.observation_space = gym.spaces.Box(low=0, high=255,
        shape=newshape)
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=qmodel,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        preobs = env.reset()
        obs = preprocess(preobs)
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_preobs, rew, done, _ = env.step(action)
            new_obs = preprocess(new_preobs)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
