from collections import deque
import random

class ReplayMemmory():
    #creat the FIFO queue=>experience replay
    def __init__(self,maxlen,seed=None):
        self.memory=deque([],maxlen=maxlen)
        
    def append(self,new_exp):
        self.memory.append(new_exp)
        
    def sample(self,sample_size):
        return random.sample(self.memory,sample_size)
    #above code is say that we want the randome sample from memmory of "sample_size"
    def __len__(self):
        return len(self.memory)
