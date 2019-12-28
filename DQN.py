import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import gym
import time


class DeepQNet(nn.Module):
    def __init__(self, dim_obs, num_act, hid_size):
        super(DeepQNet, self).__init__()

        self.dim_obs = dim_obs
        self.num_act = num_act
        self.hid_size = hid_size

        self.nn1 = nn.Linear(self.dim_obs, self.hid_size)
        self.nn2 = nn.Linear(self.hid_size, self.hid_size)
        self.nn3 = nn.Linear(self.hid_size, self.num_act)

        return

    def forward(self, obs):
        out = self.nn1(obs)
        out = F.relu(out)
        out = self.nn2(out)
        out = F.relu(out)
        out = self.nn3(out)
        return out


class Memory:
    def __init__(self, dim_obs, dim_act, max_size):
        self.max_size = max_size
        self.if_full = False
        self.index = 0

        self.obs_arr = np.zeros((max_size, dim_obs))
        self.obs_ne_arr= np.zeros((max_size, dim_obs))
        self.act_arr = np.zeros((max_size, dim_act))
        self.rwd_arr = np.zeros((max_size, 1))
        self.done_arr = np.zeros((max_size, 1), dtype=bool)
        return

    def push(self, obs, act, obs_ne, rwd, done):
        self.obs_arr[self.index] = obs
        self.obs_ne_arr[self.index] = obs_ne
        self.act_arr[self.index] = act
        self.rwd_arr[self.index] = rwd
        self.done_arr[self.index] = done

        self.index += 1
        if self.index >= self.max_size:
            self.index = 0
            self.if_full = True
        return

    def sample(self, num=1):
        max_ind = self.max_size if self.if_full else self.index
        sample_indices = np.random.choice(range(max_ind), num, replace=False)
        sample_indices = sample_indices.flatten()

        obs = self.obs_arr[sample_indices]
        act = self.act_arr[sample_indices]
        obs_ne = self.obs_ne_arr[sample_indices]
        rwd = self.rwd_arr[sample_indices]
        done = self.done_arr[sample_indices]
        return obs, act, obs_ne, rwd, done


class DQNMethod:
    def __init__(self, env, hid_net_size=64, discount_factor=0.99,
                 lr=1e-2, max_mem_size=5000, log_folder="./runs/",
                 eps_start=0.05, eps_end=0.9, eps_decay=10000):

        self.env = env

        dim_obs = env.observation_space.sample().shape[0]
        num_act = env.action_space.n

        self.num_act = num_act
        self.dim_obs = dim_obs
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.log_folder = log_folder

        self.discount_factor = discount_factor
        self.max_mem_size = max_mem_size
        self.step_count = 0
        self.threshold = 0

        self.q_net = DeepQNet(dim_obs=dim_obs, num_act=num_act, hid_size=hid_net_size)
        self.target_q_net = DeepQNet(dim_obs=dim_obs, num_act=num_act, hid_size=hid_net_size)

        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.memory = Memory(dim_obs, 1, max_mem_size)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.writer = SummaryWriter(self.log_folder+"/DQN/"
                                    +time.strftime("%b%d_%H_%M", time.localtime()))
        return

    def select_act_no_grad(self, obs, method="ep_greedy"):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float)

        if obs.ndim == 1:
            obs = obs.view(1, -1)

        with torch.no_grad():
            if method == "greedy":
                q_sa = self.q_net(obs)
                _, act = torch.max(q_sa, dim=1)
            elif method == "random":
                act = torch.randint(0, self.num_act, (obs.shape[0],))
            elif method == "ep_greedy":

                prob = np.random.uniform(0, 1)
                if prob > self.threshold:
                    act = torch.randint(0, self.num_act, (obs.shape[0],))
                else:
                    q_sa = self.q_net(obs)
                    _, act = torch.max(q_sa, dim=1)

        act = act.numpy()
        act = act.item() if act.size == 1 else act
        return act

    def train(self, max_step, training_batch=100, target_update=10, eval_interval=50):
        self.writer.add_graph(self.q_net, torch.rand(2, self.dim_obs))

        self.step_count = 0

        obs = self.env.reset()
        while self.step_count < max_step:
            self.step_count += 1

            act = self.select_act_no_grad(obs, method="ep_greedy")
            obs_ne, rwd, done, _ = self.env.step(act)
            self.memory.push(obs, act, obs_ne, rwd, done)
            obs = obs_ne if not done else self.env.reset()

            if self.step_count > training_batch:
                self.train_q_net(training_batch)
            if self.step_count % eval_interval == 0:
                self.evaluation()

            if self.step_count % target_update == 0:
                self.update_target_q()

        return

    def evaluation(self):
        test_env = gym.make(self.env.spec.id)
        sum_rwd_record = []
        for i in range(15):
            rwd_list = []
            done = False
            obs = test_env.reset()
            while not done:
                act = self.select_act_no_grad(obs, method="greedy")
                obs_ne, rwd, done, _ = test_env.step(act)
                obs = obs_ne
                rwd_list.append(rwd)

            sum_rwd = np.sum(rwd_list)
            sum_rwd_record.append(sum_rwd)

        print(f"evaluation sum reward: {np.mean(sum_rwd_record)}, std: {np.std(sum_rwd_record)}")
        self.writer.add_scalar("eval/reward", np.mean(sum_rwd_record), self.step_count)
        self.writer.add_histogram("eval/rwd_dist", np.array(sum_rwd_record), self.step_count)
        return

    def train_q_net(self, training_batch):
        self.threshold = self.eps_start + (self.eps_end - self.eps_start) * \
                    (np.exp(-1 * self.step_count / self.eps_decay))
        self.writer.add_scalar("training/threshold", self.threshold, self.step_count)

        obs, act, obs_ne, rwd, done = self.memory.sample(training_batch)
        obs = torch.FloatTensor(obs)
        act = torch.LongTensor(act)
        obs_ne = torch.FloatTensor(obs_ne)
        rwd = torch.FloatTensor(rwd)
        done = torch.BoolTensor(done)
        n_done = ~done.flatten()
        with torch.no_grad():
            q_ne = self.target_q_net(obs_ne)
            q_ne_max, _ = torch.max(q_ne, dim=1)
            q_ne_max = q_ne_max.flatten() * n_done

        q_target = rwd.flatten() + self.discount_factor * q_ne_max
        q_obs = self.q_net(obs)
        q_obs = torch.gather(q_obs, dim=1, index=act.view(-1, 1))

        loss = F.mse_loss(q_target, q_obs)
        self.q_net.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"step: {self.step_count} loss {loss.item()}")
        self.writer.add_scalar("training/loss", loss, self.step_count)
        return

    def update_target_q(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        return


if __name__ == "__main__":

    cart_env = gym.make("CartPole-v1")
    dqn_method = DQNMethod(cart_env)
    dqn_method.train(50000)
    pass


