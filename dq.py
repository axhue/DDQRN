import MalmoPython
import os
import sys
import time
import random
import json
import itertools

from baselines import deepq
from baselines import logger
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule

import tensorflow as tf
import tensorflow.contrib.layers as layers
import baselines.common.tf_util as U

from IPython.display import clear_output,display
import numpy as np


class Env:
    def __init__(self,agent,shape = 3):
        self.world_state = None
        self.agent = agent
        self.my_mission_record = MalmoPython.MissionRecordSpec()
        self.episode_length = 100
        self.data = None
        self.obs_shape = shape #default is a 3x3 shape
        
        
        self.host = MalmoPython.AgentHost()
        try:
            self.host.parse( sys.argv )
        except RuntimeError as e:
            print ('ERROR:',e)
            print (self.host.getUsage())
            if self.host.receivedArgument("help"):
                print (self.host.getUsage())
                exit(0)
        
    def observe(self):
        
        self.world_state = self.host.peekWorldState()
        while self.world_state.is_mission_running and all(e.text=='{}' for e in self.world_state.observations):
            self.world_state = self.host.peekWorldState()
   
        # If there are some new observations
        if self.world_state.number_of_observations_since_last_state > 0:
            self.data = json.loads(self.world_state.observations[-1].text)
        
        state = self.world_state
            
        #data = json.loads(state.observations[-1].text)
        # get reward if detected, else reward is -1
        reward = -1
        if state.number_of_rewards_since_last_state > 0:
            reward = state.rewards[0].getValue()
    
        # reformat grid to a vector that only show the floor with blocks
        vec = []    
        for item in self.data['grid'][::-1]:
            if 'lava' in item:
                vec.append(1)
            else:
                vec.append(0)
        
        new_state = np.array(vec)
        #.reshape(self.obs_shape,
        #                                 self.obs_shape)
        
        return(reward,new_state,self.data,state) # return r,s,data,extra_info
        
    def startworld(self,world_file):
        with open('CliffWalking.xml','r') as f:
            my_mission = MalmoPython.MissionSpec(f.read(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        # Attempt to start a mission:
        max_retries = 3
        for retry in range(max_retries):
            try:
                self.host.startMission( my_mission, my_mission_record )
                sys.stdout.write("Mission Started")
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print ("Error starting mission:{}".format(e))
                    exit(1)
                else:
                    time.sleep(2)
        # Loop until mission starts:
        print ("Waiting for the mission to start ")
        self.world_state = self.host.getWorldState()
        while (not self.world_state.has_mission_begun):
            sys.stdout.write(".")
            time.sleep(0.1)
            self.world_state = self.host.getWorldState()
            for error in self.world_state.errors:
                print ("Error:",error.text)
                
        ## wait until a valid observation        
        while self.world_state.is_mission_running and all(e.text=='{}' for e in self.world_state.observations):
            self.world_state = self.host.peekWorldState()
        #populate emtpy fields for init
        self.data = json.loads(self.world_state.observations[-1].text)
        
        return self.observe()
    def quit(self):
        self.host.sendCommand('quit')
        
    def step(action):
        self.host.sendCommand(ractions[action])
        return self.observe()
    def run(self,world,epochs=0):
        # load world
        R = 0
        for i in range(epochs):
            _, s, obs,ws = self.startworld(world)
            mission_run = ws.is_mission_running
            while(mission_run):
                # get state
                time.sleep(1)
                
                # act
                send_a,a = agent.act(s)
                print("action:{}".format(send_a))
                self.agent_host.sendCommand(send_a)

                
                # observe
                #print(a)
                r,s_prime,obs,ws = self.parse_state()
                mission_run = ws.is_mission_running
                #print("sample:{}".format(a))
                self.agent.observe((s,a,r,s_prime))
                self.agent.replay()
                
                s = s_prime
                R += r
                if (not  mission_run):
                    print("episode done")
                    time.sleep(1)
                
            print("done epoch: {}".format(i))
            
simple_actions = {
    'strafe':{
        'left': 'strafe -1',
        'right': 'strafe 1'
    },
    'move':{
        'back':'move -1',
        'forward':'move 1'
    }   
}
# flatten dict of actions
ractions = []
for action_type in simple_actions.keys():
    
    for action in simple_actions[action_type]:
        ractions.append(simple_actions[action_type][action])
        
def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out
        
        
env = Env(None)
with U.make_session(2):

    replay_buffer = ReplayBuffer(10000)
    exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)
    # Initialize the parameters and copy them to the target network.

   
    episode_rewards = [0.0]
    r,s,data,ws = env.startworld('CliffWalking.xml')
    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=lambda name: U.BatchInput((9,), name=name),
        q_func=model,
        num_actions=len(ractions),
        optimizer=tf.train.AdamOptimizer(learning_rate=5e-4)
    )
    
    U.initialize()
    update_target()
    
    for t in itertools.count():
        action = act(s, update_eps=exploration.value(t))[0]
        r,s_,data,ws = env.step(action)
        
        # store transition in memory
        replay_buffer.add(s[None],action,r,s_,not ws.is_mission_running)
        
        s = s_
        
        episode_rewards[-1] += r
        
        if ws.is_mission_running is False: #mission is done
            _,s,_,ws = env.startworld('CliffWalking.xml')
            
        is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
        
        
        if is_solved:
            # Show off the result
            print('solved')
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