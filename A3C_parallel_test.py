#%%
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import torch

#%%

class Worker(mp.Process):
    def __init__(self, i, global_net):
        super(Worker, self).__init__()
        self.i = i
        self.global_net = global_net
        self.count = 0

        return

    def run(self):
        while self.count <= 3:
            self.count += 1
            data = torch.arange(2).float()
            data = data.view(-1, 2)

            # if self.i == 2: data = -data
            result = self.global_net(data)
            result = result.sum()

            self.global_net.zero_grad()
            result.backward()


            print("before", self.i, self.global_net.nn1.weight.detach().numpy())
            # print(self.global_net.nn1.bias.grad)
            with torch.no_grad():
                self.global_net.nn1.weight += self.i * self.global_net.nn1.weight.grad
            print("after", self.i, self.global_net.nn1.weight.detach().numpy())


        return


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.nn1 = nn.Linear(2, 1)
        return

    def forward(self, x) :
        out = self.nn1(x)
        return out

#%%
# mp.set_start_method("spawn")
net = Net()
net.share_memory()
#%%
workers = [Worker(1, net), Worker(2, net)]
[_w.start() for _w in workers]
[_w.join() for _w in workers]