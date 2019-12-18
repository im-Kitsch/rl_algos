#%%
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch


#%%
class PGPolicy(torch.nn.Module):
    def __init__(self, dim_obs, num_act, num_hid):
        super(PGPolicy, self).__init__()
        self.num_act = num_act
        self.dim_obs = dim_obs
        self.hid_nn1 = torch.nn.Linear(dim_obs, num_hid)
        self.hid_nn2 = torch.nn.Linear(num_hid, num_act)

        return

    def forward(self, obs):
        out = self.hid_nn1(obs)
        out = torch.nn.functional.relu(out)
        out = self.hid_nn2(out)
        return out

    def select_act(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs)
        obs = obs.view(-1, self.dim_obs)

        obs = obs.reshape(-1, self.dim_obs)
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            output = self.forward(obs)
        act = torch.argmax(output, axis=1, keepdim=True)
        act = act.numpy()
        # act = act.reshape(-1) # for box action, act must be array
        act = act.item() if act.shape == (1, 1) else act
        return act


#%%
class PGMethod:
    def __init__(self, policy, env, lr=1e-3):
        self.policy = policy
        self.env = env
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        return

    def train(self):

        return

    def rollout_reward(self, rwd):
        rwd = np.cumsum(rwd)
        return rwd

    def rollout_n(self, n_epi):
        total_record = []
        for i in range(n_epi):
            s_list, act_list, rwd_list = [], [], []
            done = False
            s = env.reset()
            while not done:
                act = self.policy.select_act(s)
                s_ne, r, done, _ = self.env.step(act)
                s_list.append(s)
                act_list.append(act)
                rwd_list.append(r)
                s = s_ne
            s_list = np.array(s_list)
            act_list = np.array(act_list)
            rwd_list = np.array(rwd_list)
            r_cum = self.rollout_reward(rwd_list)
            record = {"s": s_list, "a": act_list, "r": rwd_list, "R_tao": r_cum}
            total_record.append(record)
        return total_record


#%%

NUM_HID = 64

env = gym.make("CartPole-v1")
dim_obs = env.observation_space.sample().shape[0]
num_act = env.action_space.n

policy = PGPolicy(dim_obs, num_act, NUM_HID)
pg_method = PGMethod(policy, env)

#%%
s = env.reset()
done = False
while not done:
    act = policy.select_act(s)
    s, r, done,  _ = env.step(act)
    print(f"s {s} a {act}  r {r}")