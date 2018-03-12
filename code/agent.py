import numpy as np
from keras.models import Sequential,model_from_json,Model
#from keras.layers import Conv2D,LSTM,GRU,TimeDistributed,Dense,Flatten,Input,Lambda,multiply
from keras.optimizers import RMSprop,Adam
from keras import backend as K
import tensorflow as tf
import itertools

from keras.optimizers import (Adam, RMSprop)
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
        Permute, merge, Merge, multiply, Lambda, Reshape, TimeDistributed, LSTM, RepeatVector, Permute)
from keras.layers.wrappers import Bidirectional


from memory import Memory,ShortMemory
from networks import Network
from loss import mean_huber_loss
import random

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
K.tensorflow_backend.set_session(sess)



# There are way too many parameters for the agent to be within the init arguments
class AgentConfig():
    def __init__(self):
        '''
        stateCnt 
        actionCnt 
        mem_size
        epsilon_policy 
        gamma 
        num_frames
        learning_rate 
        train_start 
        train_freq 
        target_update_freq  
        batch_size 
        preprocessor 
        log_path 
        name      
        '''
        self.stateCnt = None
        self.actionCnt = None
        self.mem_size = None
        self.epsilon_policy = None
        self.gamma = None
        self.num_frames = None
        self.learning_rate = None
        self.train_start = None
        self.train_freq = None
        self.target_update_freq = None 
        self.batch_size = None
        self.preprocessor = None
        self.log_path = None
        self.name = None
    def validate(self):
        # validate parameters
        pass
        
class Agent:

    def __init__(self,cfg):
        self.stateCnt = cfg.stateCnt
        if cfg.preprocessor:
            self.preprocessor = cfg.preprocessor
            self.stateCnt = cfg.preprocessor.new_shape

        self.actionCnt = cfg.actionCnt



        # initialize memory
        self.longmem = Memory(cfg.mem_size,cfg.num_frames,self.stateCnt)
        self.shortmem = ShortMemory(cfg.num_frames,self.stateCnt)


        self.epsilon_policy = cfg.epsilon_policy
        self.gamma = cfg.gamma
        self.target_update_freq = cfg.target_update_freq

        self.name = cfg.name
        self.learning_rate = cfg.learning_rate 
        self.train_start = cfg.train_start
        self.train_freq = cfg.train_freq
        self.batch_size = cfg.batch_size

        # build network
        inpt = (cfg.num_frames,) + self.stateCnt
        self.net_inpt = inpt
        self.model = Network(inpt,self.actionCnt,True,'duel',cfg.learning_rate,'Q-network').build2() # model
        self.target_model = Network(inpt,self.actionCnt,True,'duel',cfg.learning_rate, 'Target-network').build2() # target model
        
        self.final_model = None
        self.compile()
        #logging
        if cfg.log_path:
            self.writer = tf.summary.FileWriter(cfg.log_path)
            self.log_path = cfg.log_path
        # init counters for logging purpose
        self.loss_count = 0
        self.counter = 0
    def save_model(self):
        self.model.save_weights(self.log_path + "/" + self.name+".h5")
    def remember(self,s,a,r,done):
        self.longmem.add(s,a,r,done)
    def save_scalar(self,step, name, value):
        """Save a scalar value to tensorboard.
          Parameters
          ----------
          step: int
            Training step (sets the position on x-axis of tensorboard graph.
          name: str
            Name of variable. Will be the name of the graph in tensorboard.
          value: float
            The value of the variable at this step.
          writer: tf.FileWriter
            The tensorboard FileWriter instance.
          """
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = float(value)
        summary_value.tag = name
        self.writer.add_summary(summary, step)
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def compile(self, optimizer = None, loss_func = None):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is the place to create the target network, setup 
        loss function and any placeholders.
        """
        if loss_func is None:
            loss_func = mean_huber_loss
            # loss_func = 'mse'
        if optimizer is None:
            optimizer = Adam(lr = self.learning_rate)
            # optimizer = RMSprop(lr=0.00025)
        with tf.variable_scope("Loss"):
            state = Input(shape = (*self.net_inpt, ) , name = "states")
            action_mask = Input(shape = (self.actionCnt,), name = "actions")
            qa_value = self.model(state)
            qa_value = multiply([qa_value, action_mask], name = "multiply")
            qa_value = Lambda(lambda x: tf.reduce_sum(x, axis=1, keep_dims = True), name = "sum")(qa_value)

        self.final_model = Model(inputs = [state, action_mask], outputs = qa_value)
        self.final_model.compile(loss=loss_func, optimizer=optimizer)

    def act(self, s):
        # expect s.shape = (frames,height,width,depth)
        # Epsilon greedy action selection
        s = s[None] # increase the rank of tensor to have a batch_size of 1 and length 1

            
        if np.random.rand() <= self.epsilon_policy.epsilon:
            return random.randrange(self.actionCnt)
        else:
            act_values = self.model.predict_on_batch(s)
            self.epsilon_policy.update()
        return np.argmax(act_values[0]) # returns action
    def replay(self,batch_size):
        prebatch_s,batch_a,batch_r,prebatch_s_,batch_done = self.longmem.sample(batch_size)# a batch of episode of parameter length

        batch_s = self.preprocessor.process_batch(prebatch_s)
        batch_s_ = self.preprocessor.process_batch(prebatch_s_)

        a_ = self.model.predict(batch_s_)
        a_idx = np.argmax(a_,axis=1)
        behaviour_q = self.target_model.predict(batch_s_)
        action_mask = np.zeros((batch_size, self.actionCnt))
        action_mask[range(batch_size), batch_a] = 1.0
        
        done_mask = np.array([1-x for x in batch_done]) # update target with bootstrap on non terminal states else terminal states are updates with just the reward
        #target = self.model.predict(batch_s) [range(batch_size),batch_a.astype('int')] # smaller update for other residuals? research if this actually makes in impact?
        
        target = batch_r + self.gamma *done_mask*(behaviour_q[range(batch_size),a_idx]) 
        loss = self.final_model.train_on_batch([batch_s,action_mask],target)

        return loss, np.mean(target)
    def init_loop(self,env):
        print("Filling memory with {} episodes".format(self.train_start))
        # intializing loop to fill memory
        for i in range(self.train_start):
            pre_s = env.reset()
            done = False
            self.shortmem.forget() # forget short term memory for recurrent network 
            
            for t in itertools.count():
                
                mem_s = self.preprocessor.process_state_for_memory(pre_s) #scaled and grayscaled
                net_s = self.preprocessor.process_for_network(mem_s) # normalized
                
                self.shortmem.add(net_s)
                
                hist_s = self.shortmem.get() # receive the last frames including the most recent frame to make a tensor shape (num_frames,frame_dim)
                
                a = env.action_space.sample()
                
                pre_s_, raw_r, done, info = env.step(a)
                
                r = self.preprocessor.process_reward(raw_r) # reward clipping
                mem_s_ = self.preprocessor.process_state_for_memory(pre_s_) #scaled and grayscaled
                net_s_ = self.preprocessor.process_for_network(mem_s_) # normalized

                #self.shortmem.add(net_s_) with this we get  double frames
                self.remember(mem_s,a,r,done)
                
                pre_s = pre_s_
                
                if done:
                    break
        print("Done filling memory")
        
    def train(self,env,episodes,render = False):
        
        
        counter = 0
        
        self.init_loop(env)
        
        # play loop
        print("Training")
        for e in range(episodes):
            # stats
            e_loss = .0
            e_length = .0
            e_r = .0
            e_raw_r = .0
            e_target_val = .0


            #agent.model.reset_states()
            pre_s = env.reset()
            done = False
            self.shortmem.forget() # forget short term memory for recurrent network 
            
            if e % 40 == 0:
                self.save_model()
                
            for t in itertools.count():
                mem_s = self.preprocessor.process_state_for_memory(pre_s) #scaled and grayscaled
                net_s = self.preprocessor.process_for_network(mem_s) # normalized
                
                self.shortmem.add(net_s)
                
                hist_s = self.shortmem.get() # receive the last frames including the most recent frame to make a tensor shape (num_frames,frame_dim)
                # take action using net_s and receive a
                a = self.act(hist_s)
                if render:
                    env.render()

                pre_s_, raw_r, done, info = env.step(a)

                r = self.preprocessor.process_reward(raw_r) # reward clipping

                self.shortmem.add(net_s_) #with this we get  double frames
                self.remember(mem_s,a,r,done)
            
                # collect stats
                e_raw_r += raw_r
                e_r += r
                
                counter +=1 

                if counter % (self.train_freq * self.target_update_freq) == 0:
                    self.update_target_model()
                if counter % self.train_freq == 0  :
                    loss, avg_target = self.replay(self.batch_size)
                    e_loss += loss
                    e_target_val += avg_target

                      
                pre_s = pre_s_
                if done:
                    #capture last frame
                    #last_frame = self.preprocessor.process_state_for_memory(pre_s_)
                    #self.remember(last_frame,a,0,done)
                    
                    #broadcast stats
                    print("Training: episode %d, length %d, reward %.0f, raw_reward %.0f, loss %.4f, target value %.4f,, memory cap %d" % 
                        (e, t, e_r, e_raw_r, e_loss, e_target_val/t, self.longmem.current))
                    self.save_scalar(e,"train/episode_reward",e_r)
                    self.save_scalar(e,"train/episode_raw_reward",e_raw_r)
                    self.save_scalar(e,"train/episode_frames",t)
                    self.save_scalar(e,"train/episode_loss",e_loss)
                    self.save_scalar(e,"train_avg/avg_target_value",e_target_val/t)
                    self.save_scalar(e, 'train_avg/avg_reward', e_r / t)
                    self.save_scalar(e, 'train_avg/avg_loss', e_loss / t)
                    break