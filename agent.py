import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemmory
import itertools
import torch
import torch.nn as nn
import yaml
import torch.optim as optim
import os
import argparse
import random

if torch.backends.mps.is_available():
    device="mps"
elif torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
    
RUNS_DIR="runs"#folder name
os.makedirs(RUNS_DIR,exist_ok=True)#this create the folder {RUNS_DIR} if it is not exist then it will created
class Agent:
    def __init__(self,param_set):
        self.param_set=param_set
        
        with open("parameter.yaml","r") as f:
            all_param_set=yaml.safe_load(f)
            params=all_param_set[param_set]
            
        self.alpha=params["alpha"]
        self.gamma=params["gamma"]
        self.epsilon_init=params["epsilon_init"]
        self.epsilon_min=params["epsilon_min"]
        self.epsilon_decay=params["epsilon_decay"]
        
        self.replay_memory_size=params["replay_memory_size"]
        self.min_batch_size=params["min_batch_size"]
        
        self.reward_threshold=params["reward_threshold"]
        self.min_batch_size=params["min_batch_size"]
        self.network_sync_rate=params["network_sync_rate"]
        
        self.loss_fn=nn.MSELoss()
        self.optimizer=None
        
        self.LOG_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.log")
        self.MODEL_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.pt")
        
    def run(self,is_training=True,render=False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)
        num_state=env.observation_space.shape[0]#input dim
        num_actions = env.action_space.n # output dim
        
        policy_dqn=DQN(num_state,num_actions).to(device)
        
        if is_training:
            memory=ReplayMemmory(self.replay_memory_size) #maxlen=10000
            epsilon=self.epsilon_init
            
            target_dqn=DQN(num_state,num_actions).to(device)
            #copy the wt & bias values from the policy network => target_dqn
            target_dqn.load_state_dict(policy_dqn.state_dict())#state_dict()= in this all state values are store
            steps=0
            self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)
            best_reward=float("-inf")
        else:
            #in testing state
            #load beat model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
        for episode in itertools.count():
            state, _ = env.reset()
            state=torch.tensor(state,dtype=torch.float,device=device)#convert the state values in tensor
            
            episode_rewards=0
            terminated=False
            while (not terminated and episode_rewards<self.reward_threshold):#this is one 
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random()<epsilon:
                    action = env.action_space.sample()#explore
                    action=torch.tensor(action,dtype=torch.long,device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()#expoit
                # Processing:
                next_state, reward, terminated, _,_ = env.step(action.item())
                #creating tensors
                reward=torch.tensor(reward,dtype=torch.float,device=device)
                next_state=torch.tensor(next_state,dtype=torch.float,device=device)
                
                if is_training: #add new experience
                    memory.append((state,action,next_state, reward, terminated))
                    steps+=1
                state=next_state
                episode_rewards+=reward.item()
            print(f"Episode:{episode+1} and episode rewards:{episode_rewards} epsilon:{epsilon}")
            #epsilon deacy
            if is_training:
                epsilon=max(epsilon*self.epsilon_decay,self.epsilon_min)
                if episode_rewards>best_reward:
                    log_msg=f"best reward:{best_reward} for episode:{episode+1}"
                    with open(self.LOG_FILE,"a") as f:
                        f.write(log_msg+"\n")
                        
                    torch.save(policy_dqn.state_dict(),self.MODEL_FILE)
                    best_reward=episode_rewards
            if is_training and len(memory)>self.min_batch_size:
                #get sample
                mini_batch=memory.sample(self.min_batch_size)
                self.optimize(mini_batch,policy_dqn,target_dqn)
                
                #sync network
                if steps>self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps=0
                
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        #get experiences
        states,actions,next_states,rewards,termiantes=zip(*mini_batch)
        
        states=torch.stack(states)
        actions=torch.stack(actions)
        next_states=torch.stack(next_states)
        rewards=torch.stack(rewards)
        termiantes=torch.tensor(termiantes,dtype=torch.float,device=device)
        #predict Q va;ues for the actions your agent took
        current_q=policy_dqn(states).gather(dim=1,index=actions.unsqueeze(1)).squeeze(1)
        #calculate targets safely with standerd vectorization
        with torch.no_grad():
            max_next_q=target_dqn(next_states).max(dim=1)[0]
            target_q=rewards+(self.gamma*max_next_q*(1-termiantes))
        #loss optimizize network weights
        loss=self.loss_fn(current_q,target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
if __name__=="__main__":
    #parse command line inputs
    parser=argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters",help="")
    parser.add_argument("--train",help="Trainig mode",action="store_true")
    args=parser.parse_args()
    
    dql=Agent(param_set=args.hyperparameters)
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False,render=True)
            
    #env.close()=>we can stop it manually stop
