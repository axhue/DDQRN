from code.agent import Agent,AgentConfig
import gym
from code.preprocessors import AtariPreprocessor
from code.policies import LinearDecayGreedyEpsilonPolicy
from code.utils import get_output_folder
import os
from keras import backend as K
import tensorflow as tf
import numpy as np




if __name__== '__main__':
    atari_env = gym.make('Pitfall-v0')
    doc = get_output_folder(None,"./logs","Pitfall-v0","DDQRN20")
    #np.random.seed(0)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)
    
    episodes = 100000
    cfg = AgentConfig()
    cfg.stateCnt = atari_env.observation_space.shape
    cfg.actionCnt = atari_env.action_space.n
    cfg.mem_size = 1000000
    cfg.epsilon_policy = LinearDecayGreedyEpsilonPolicy(1.0,0.05,episodes+300000 )
    cfg.gamma = 0.99
    cfg.num_frames = 10
    cfg.learning_rate = 0.0001
    cfg.train_start = 50
    cfg.train_freq = 4
    cfg.target_update_freq = 10000
    cfg.batch_size = 32
    cfg.preprocessor = AtariPreprocessor((84,84,1))
    cfg.log_path = doc
    cfg.name = "R20"
    agent = Agent(cfg,sess)
    agent.train(atari_env,episodes,render=False)
    #agent.fit(atari_env,episodes)