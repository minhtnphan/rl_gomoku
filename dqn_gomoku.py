from gomoku import RandomGomoku
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# This code is understood and transferred from
# https://nbviewer.org/github/Curt-Park/rainbow-is-all-you-need/blob/master/01.dqn.ipynb

class ReplayBuffer:
    def __init__(self, obs_dim, size, batch_size=32):
        self.obs_buf = np.zeros([size] + list(obs_dim), dtype=np.int8)
        self.next_obs_buf = np.zeros([size] + list(obs_dim), dtype=np.int8)
        self.acts_buf = np.zeros([size], dtype=np.int32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size = size
        self.batch_size = batch_size
        self.ptr, self.size = 0, 0

    def store(self, obs, act, next_obs, reward, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.next_obs_buf[self.ptr] = next_obs
        self.rews_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size  # Reset when buffer is full
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs], acts=self.acts_buf[idxs], next_obs=self.next_obs_buf[idxs],
                    rews=self.rews_buf[idxs], done=self.done_buf[idxs])

    def __len__(self):
        return self.size

class NetWork(nn.Module):
    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size
        self.fc1 = nn.Linear(board_size[0] * board_size[1], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, board_size[0] * board_size[1])

    def forward(self, x):
        x = x.view(-1, self.board_size[0] * self.board_size[1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x


class CNN(nn.Module):
    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (self.board_size[0] - 4) ** 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, board_size[0] * board_size[1])

    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        x = F.relu(self.conv2(x))
        # x = self.pool2(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

class DQNAgent:
    def __init__(self, env: gym.Env, memory_size: int, batch_size: int, target_update: int,
                 epsilon_decay: float, seed: int = 0, max_epsilon: float = 1.0,
                 min_epsilon: float = 0.01, gamma: float = 0.95):
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.n
        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.seed = seed
        self.target_update = target_update
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.dqn = NetWork(obs_dim).to(self.device)
        self.dqn_target = NetWork(obs_dim).to(self.device)  # Initialize target network != policy network
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()  # Set target network to calculate only, not train with SGD
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=1e-4)  # Optimizer on policy network parameters

        self.transition = list()
        self.is_test = False  # To define mode of train / test

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if not self.is_test:

            self.transition += [next_state, reward, done]
            self.memory.store(*self.transition)
        return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch()
        loss = self._compute_dqn_loss(samples)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, num_step: int, plotting_interval: int = 1000):
        self.is_test = False
        state, _ = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        average_losses = []
        average_scores = []
        average_epsilons = []
        score = 0

        for step in range(1, num_step+1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward
            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                self.epsilon = max(self.min_epsilon, self.epsilon -
                                   (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
                epsilons.append(self.epsilon)
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()
            if step % plotting_interval == 0:
                average_losses.append(sum(losses[-plotting_interval:]) / plotting_interval)
                average_scores.append(sum(scores[-plotting_interval:]) / plotting_interval)
                average_epsilons.append(sum(epsilons[-plotting_interval:]) / plotting_interval)
                # self._plot(step, scores, losses, epsilons)
                self._plot(step, average_scores, average_losses, average_epsilons)
        self.env.close()
    def test(self, video_folder: str):
        self.is_test = True
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        state, _ = self.env.reset()
        done = False
        score = 0
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward
        print("Score: ", score)
        self.env.close()
        self.env = naive_env

    def run(self, num_episodes, render=False):
        self.is_test = True
        self.epsilon = 0
        rewards = []
        for episode in range(num_episodes):

            obs, _ = self.env.reset()
            done = False
            while not done:
                action = self.select_action(obs)
                new_obs, reward, done = self.step(action)
                obs = new_obs
                if render:
                    self.env.render()


    def _compute_dqn_loss(self, samples):
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(device)
        loss = F.smooth_l1_loss(curr_q_value, target)
        return loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(self, frame_idx, scores, losses, epsilons):
        """Plot the training progresses."""
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.pause(5)
        plt.close()


seed = 1

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
seed_torch(seed)
# parameters
num_frames = 1000000
memory_size = 1000
batch_size = 32
target_update = 100
epsilon_decay = 3 / num_frames

env = RandomGomoku(size=19, render_mode="human")
agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, seed)
agent.train(num_frames)
agent.run(num_episodes=10, render=True)
video_folder = "videos/dqn"
agent.test(video_folder=video_folder)
