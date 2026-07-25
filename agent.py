import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experiwnce_replay import ReplayMemmory

if torch.backend.mps.is_availabel():
    device="mps"
elif torch.cuda.is_availabel():
    device="cuda"
else:
    device="cpu"
def run(self,is_training=True,reder=False):
    env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)
    num_state=env.observation_space.shape[0]#input dim
    c=env.observation_space.shape[1]#input dim
    
    policy_dqn=DQN(num_state,num_state).to(device)
    state, _ = env.reset()
    
    if is_training:
        memory=ReplayMemmory(10000) #maxlen=10000
        
    while True:
        # Next action:
        # (feed the observation to your agent here)
        action = env.action_space.sample()

        # Processing:
        next_state, reward, terminated, _,_ = env.step(action)
        
        if is_training: #add new experience
            memory.append((state,action,next_state, reward, terminated))
        
        # Checking if the player is still alive
        if terminated:
            break

    env.close()
