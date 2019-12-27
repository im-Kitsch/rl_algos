#%%
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
import time

#TODO  better evaluation; test file; debugging, stop criterion

#%%
class PGPolicy(torch.nn.Module):
    def __init__(self, dim_obs, num_act, num_hid):
        super(PGPolicy, self).__init__()
        self.num_act = num_act
        self.dim_obs = dim_obs
        self.hid_nn1 = torch.nn.Linear(dim_obs, num_hid)
        self.hid_nn2 = torch.nn.Linear(num_hid, num_hid)
        self.hid_nn3 = torch.nn.Linear(num_hid, num_act)

        return

    def forward(self, obs):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        out = self.hid_nn1(obs)
        out = torch.nn.functional.relu(out)
        out = self.hid_nn2(out)
        out = torch.nn.functional.relu(out)
        out = self.hid_nn3(out)
        out = torch.nn.functional.log_softmax(out, dim=1)
        return out

    def select_act(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs)
        obs = obs.view(-1, self.dim_obs)

        obs = obs.reshape(-1, self.dim_obs)
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            log_prob = self.forward(obs)
            sampler = torch.distributions.Categorical(logits=log_prob)
            act = sampler.sample()

        act = act.numpy()
        # act = act.reshape(-1) # for box action, act must be array

        act = act.item() if act.size == 1 else act
        return act


#%%
class PGMethod:
    def __init__(self, policy, env, lr=1e-2, discount=None,
                 use_baseline=False, log_folder="./runs/"):
        self.policy = policy
        self.env = env
        self.lr = lr
        self.discount = discount
        self.use_baseline = use_baseline

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.record_rwd = []
        self.log_folder = log_folder
        self.writer = SummaryWriter(self.log_folder+"/PG_"
                                    +time.strftime("%b%d_%H_%M", time.localtime()))
        return

    def record_and_report(self, records):
        for _record in records:
            rwds_list = _record["r"].sum()
        self.record_rwd.append(np.mean(rwds_list))
        return

    def train(self, training_epoch=100, sampling_epoch=5):
        last_best = 400
        self.writer.add_graph(self.policy, torch.rand(1, 4))

        for i in range(training_epoch):
            records = self.rollout_n(sampling_epoch)
            total_loss = 0

            rwd_sum_reco = []

            if self.use_baseline:
                baseline = [item["r"].sum() for item in records]
                baseline = np.mean(baseline)
            else:
                baseline = 0

            for _reco in records:
                s, a, r, R_tao = _reco["s"], _reco["a"], _reco["r"], _reco["R_tao"],
                s = torch.FloatTensor(s)
                a = torch.LongTensor(a)
                R_tao = torch.FloatTensor(R_tao - baseline)

                log_prob = self.policy(s)
                log_prob = torch.gather(log_prob, dim=1, index=a.view(-1, 1))

                log_prob = log_prob.view(-1)
                loss = -log_prob * R_tao
                loss = loss.sum()
                total_loss += loss

                rwd_sum_reco.append(r.sum())

            print(f"epoch {i}, sampled {len(records)},"
                  f" total reward mean {np.mean(rwd_sum_reco)}, "
                  f"std {np.std(rwd_sum_reco)}")

            total_loss /= len(records)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.record_and_report(records)
            self.writer.add_scalar("loss", total_loss, i)
            self.writer.add_scalar("reward", np.mean(rwd_sum_reco), i)
            self.writer.add_histogram("reward_errbar", np.array(rwd_sum_reco), i)
            self.writer.flush()

            if np.mean(rwd_sum_reco) > last_best:
                eval_records = self.rollout_n(60)
                eval_rwd_sum = [np.sum(_record["r"]) for _record in eval_records]

                if np.mean(eval_rwd_sum) > last_best:
                    last_best = np.mean(eval_rwd_sum)

                print(f"epoch {i} evaluation: {np.mean(eval_rwd_sum)} the best {last_best}")
                if last_best == 500:
                    print("finish training")
                    break

        self.writer.close()
        plt.plot(self.record_rwd)
        plt.show()
        return

    def rollout_reward(self, rwd):
        if self.discount is None:
            return self.rollout_reward_no_discount(rwd)
        else:
            return self.rollout_reward_discount(rwd)

    def rollout_reward_no_discount(self, rwd):
        rwd = np.cumsum(rwd[::-1])[::-1]
        return rwd.copy()  #copy() is important

    def rollout_reward_discount(self, rwd):
        rwd = np.flip(rwd)
        R_tao = np.zeros_like(rwd)
        cache = 0.
        for i, _rwd in enumerate(rwd):
            cache = _rwd + self.discount * cache
            R_tao[i] = cache
        R_tao = np.flip(R_tao)
        return R_tao.copy()

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
DISCOUNT_FACTOR = 0.9999
USE_BASELINE = True
# LOG_FOLDER = "/tmp/pg_log/"
LOG_FOLDER = "./vpg/log"


torch.manual_seed(0)
np.random.seed(0)

env = gym.make("CartPole-v1")
env.seed(0)

dim_obs = env.observation_space.sample().shape[0]
num_act = env.action_space.n

policy = PGPolicy(dim_obs, num_act, NUM_HID)
pg_method = PGMethod(policy, env, use_baseline=USE_BASELINE, discount=DISCOUNT_FACTOR, log_folder=LOG_FOLDER)

#%%
pg_method.train(training_epoch=8000, sampling_epoch=15)