import numpy as np
class Memory():
    def __init__(self,capacity,hist_len,s_dim):
        '''
        capacity: how many episodes to store?
        hist_len: what is the history length of each episode?
        s_dim: the size of your state in a tuple ex. (80,80,1) 
        '''
        self.memory_size = capacity
        self.history_length = hist_len
        self.state_dim = s_dim
        self.mem_a = np.zeros(self.memory_size, dtype = np.int8)
        self.mem_r = np.zeros(self.memory_size, dtype = np.int8)
        self.mem_s = np.zeros((self.memory_size,) + s_dim , dtype = np.uint8)
        self.dones = np.zeros(self.memory_size, dtype = np.bool)
        self.current = 0
    def get_state(self,idx):
        state = self.mem_s[idx - self.history_length + 1:idx + 1, :, :]
        assert len(state) <= self.history_length
        #print(len(state))
        if len(state) < self.history_length:
            pad = self.history_length - len(state)
            pad_shape = (pad,) + self.state_dim
            #print("pad {}".format(pad_shape))
            pad_arr = np.zeros((pad,) + self.state_dim)

            state = np.concatenate((pad_arr,state),axis=0)
            #print(state.shape)

        return state
    def add(self,s,a,r,done):
        self.mem_a[self.current % self.memory_size] = a
        self.mem_r[self.current % self.memory_size] = r
        self.mem_s[self.current % self.memory_size] = s
        self.dones[self.current % self.memory_size] = done
        self.current += 1 
    def sample(self, batch_size):
        indexes = []
        # ensure enough frames to sample
        assert self.current > self.history_length
        # -1 because still need next frame
        end = min(self.current, self.memory_size) - 1

        while len(indexes) < batch_size: 
            index = np.random.randint(self.history_length - 1, end)
            # sampled state shouldn't contain episode end
            if self.dones[index - self.history_length + 1: index + 1].any():
                continue
            indexes.append(index)

        smp_s = []
        smp_a = [] 
        smp_r = []
        smp_s_ = []
        smp_done = []
        for idx in indexes:
            smp_s.append(self.get_state(idx))
            smp_a.append(self.mem_a[idx])
            smp_r.append(self.mem_r[idx])
            smp_s_.append(self.get_state(idx + 1))
            smp_done.append(self.dones[idx])
        return np.array(smp_s),np.array(smp_a),np.array(smp_r),np.array(smp_s_),np.array(smp_done)
        
class ShortMemory():
    def __init__(self,hist_len,state_dim):
        self.history_length = hist_len
        self.state_dim = state_dim
        self.mem_hist = np.zeros((hist_len,) + state_dim , dtype = np.float32)
        self.counter = 0
    def add(self,state):
        if len(self.mem_hist) == self.history_length:
            self.mem_hist = np.roll(self.mem_hist,-1,axis=0)
            self.mem_hist[-1] = state
        else:
            self.mem_hist[self.counter % self.history_length] = state
            self.counter += 1

    def get(self):
        '''
        This function provides the recent history of length history_length.
        The sample in the beginning will be padded at the beginning. (0,0,0..data)
        '''
        return self.mem_hist
    
    def forget(self):
        self.counter = 0
        self.mem_hist = np.zeros((self.history_length,) + self.state_dim, dtype = np.float32)
    