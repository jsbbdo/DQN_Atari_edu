import torch
import torch.nn as nn 
import numpy as np
import yaml

from replay_buffer import ReplayBuffer
from utils_func import make_env, record_video, play_video, smoothen, epsilon_schedule, play_and_record, evaluate

import copy
import matplotlib.pyplot as plt
from tqdm import trange
import os
import argparse

from IPython.display import clear_output
from datetime import datetime

os.makedirs("./weights", exist_ok=True)
with open('./config.yml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, env, epsilon=0):
        self.config = config
            
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.network = nn.Sequential()
        self.network.add_module('conv1', nn.Conv2d(4,16,kernel_size=8, stride=4))
        self.network.add_module('relu1', nn.ReLU())
        self.network.add_module('conv2', nn.Conv2d(16,32,kernel_size=4, stride=2))
        self.network.add_module('relu2', nn.ReLU())
        self.network.add_module('flatten', nn.Flatten())
        self.network.add_module('linear3', nn.Linear(2592, 256)) #2592 calculated above
        self.network.add_module('relu3', nn.ReLU())
        self.network.add_module('linear4', nn.Linear(256, n_actions))

        self.env = env
        self.env_name = config["env_name"]
        
    def forward(self, state_t):
        # pass the state at time t through the newrok to get Q(s,a)
        qvalues = self.network(state_t)
        return qvalues

    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        states = torch.tensor(states, device=device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
    
    def compute_td_loss(agent, use_ddqn, target_network, states, actions, rewards, next_states, done,
                    gamma=0.99, device=device):
        
        # convert numpy array to torch tensors
        states = torch.tensor(states, device=device, dtype=torch.float)
        actions = torch.tensor(actions, device=device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=device, dtype=torch.float)
        done = torch.tensor(done.astype('float32'),device=device,dtype=torch.float)
        
        q_s = agent(states)
        q_s_a = q_s[range(len(actions)), actions]
        q_s_target_next = target_network(next_states)
        if use_ddqn == True:
            q_s_next = agent(next_states).detach()
            # compute Q argmax(next_states, actions) using predicted next q-values
            _, almax = torch.max(q_s_next, dim=1)
            #use target network to calculate the q value for best action chosen above
            q_s_a_next = q_s_target_next[range(len(almax)), almax]
            # compute "td_target"
            td_target = rewards + gamma * q_s_a_next * (1-done)
            loss = torch.mean((q_s_a - td_target).pow(2))
        else:
            q_s_a_next, _ = torch.max(q_s_target_next, dim=1)
            # compute "td_target"
            td_target = rewards + gamma * q_s_a_next * (1-done)    
            loss = torch.mean((q_s_a -td_target.detach()).pow(2))
    
        return loss
    
    def train(self):
    
        mean_rw_history = []
        td_loss_history = []
        
        timesteps_per_epoch = self.config["timesteps_per_epoch"]
        batch_size = self.config["batch_size"]
        total_steps = self.config["total_steps"]
        start_epsilon = self.config["start_epsilon"]
        end_epsilon = self.config["end_epsilon"]
        eps_decay_final_step = self.config["eps_decay_final_step"]
        loss_freq = self.config["loss_freq"]
        refresh_target_network_freq = self.config["refresh_target_network_freq"] 
        eval_freq = self.config["eval_freq"]
        max_grad_norm = self.config["max_grad_norm"]
        
        state, _ = self.env.reset(seed=127)
        
        agent = self.network.to(device)
        weights_file = f'./weights/checkpoint__{self.env_name}__.pt'
        if os.path.exists(weights_file):
            agent.load_state_dict(torch.load(weights_file, weights_only=True))
        target_network = copy.deepcopy(agent)
        
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

        exp_replay = ReplayBuffer(self.config["replay_buffer_size"])

        for step in trange(total_steps + 1):

            # reduce exploration as we progress
            self.epsilon = epsilon_schedule(start_epsilon, end_epsilon, step, eps_decay_final_step)

            # take timesteps_per_epoch and update experience replay buffer
            _, state = play_and_record(state, self, env, exp_replay, timesteps_per_epoch)

            # train by sampling batch_size of data from experience replay
            states, actions, rewards, next_states, done_flags = exp_replay.sample(batch_size)


            # loss = <compute TD loss>
            loss = self.compute_td_loss(self, target_network,
                                         states, actions, rewards, next_states, done_flags,
                                         gamma=0.99,
                                         device=device)
            
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            self.optimizer.step()


            if step % loss_freq == 0:
                td_loss_history.append(loss.data.cpu().item())

            if step % refresh_target_network_freq == 0:
                # Load agent weights into target_network
                target_network.load_state_dict(agent.state_dict())

            if step % eval_freq == 0:
                # eval the agent
                mean_rw_history.append(evaluate(
                    self.env, self, n_games=3, greedy=True, t_max=1000)
                )

                clear_output(True)
                print("buffer size = %i, epsilon = %.5f" %
                    (len(exp_replay), self.epsilon))

                plt.figure(figsize=[16, 5])
                plt.subplot(1, 2, 1)
                plt.title("Mean return per episode")
                plt.plot(mean_rw_history)
                plt.grid()

                assert not np.isnan(td_loss_history[-1])
                plt.subplot(1, 2, 2)
                plt.title("TD loss history (smoothened)")
                plt.plot(smoothen(td_loss_history))
                plt.grid()

                plt.show()
                
        torch.save(agent.state_dict(), "./weights/checkpoint__{}__.pt".format(self.env_name))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train or Play.')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()
    
    env = make_env(config["env_name"])
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    if args.train:
        Agent = DQNAgent(state_dim, n_actions, env, epsilon=1)
        Agent.train()
    else:
        Agent = DQNAgent(state_dim, n_actions, env)
        weights_file = f'./weights/checkpoint__{config["env_name"]}__.pt'
        Agent.network.load_state_dict(torch.load(weights_file, weights_only=True))