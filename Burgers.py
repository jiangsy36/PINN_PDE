#1-D burgers equation
#\partial u/partial t + u \partial u/ \partial x = \nu \partial^2 u/\partial x^2

import math
import seaborn as sns
import tqdm as tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许多个 OpenMP 运行时

torch.manual_seed(42)
np.random.seed(42)


htest = 0.1
x_test = torch.arange(-1,1+htest,htest)
ktest = 0.1
t_test = torch.arange(0,1+ktest,ktest)
#print(torch.stack(torch.meshgrid(x_test,t_test)).reshape(2,-1))
X_test = torch.stack(torch.meshgrid(x_test,t_test)).reshape(2,-1).T
#print(X_test)
#Boundary conditions
bc1 = torch.stack(torch.meshgrid(x_test[0], t_test)).reshape(2, -1).T
bc2 = torch.stack(torch.meshgrid(x_test[-1], t_test)).reshape(2, -1).T

#Initial conditions
ic = torch.stack(torch.meshgrid(x_test, t_test[0])).reshape(2, -1).T
y_bc1 = torch.zeros(len(bc1))
print(len(bc1))
y_bc2 = torch.zeros(len(bc2))
y_ic = -torch.sin(math.pi * ic[:, 0])
print(torch.cat([bc1, bc2, ic]).shape)
"""
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.net(x)


class Net:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NN().to(device)

        self.h = 0.1  #space step
        self.k = 0.1  #time step
        x = torch.arange(-1, 1 + self.h, self.h)
        t = torch.arange(0, 1 + self.k, self.k)
        self.X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T

        #Boundary conditions
        bc1 = torch.stack(torch.meshgrid(x[0], t)).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(x[-1], t)).reshape(2, -1).T

        #Initial conditions
        ic = torch.stack(torch.meshgrid(x, t[0])).reshape(2, -1).T
        self.X_train = torch.cat([bc1, bc2, ic])  #cat means concatenate

        y_bc1 = torch.zeros(len(bc1))
        y_bc2 = torch.zeros(len(bc2))
        y_ic = -torch.sin(math.pi * ic[:, 0])
        self.y_train = torch.cat([y_bc1, y_bc2, y_ic])
        self.y_train = self.y_train.unsqueeze(1)  #变成张量

        self.X = self.X.to(device)
        self.y_train = self.y_train.to(device)
        self.X_train = self.X_train.to(device)

        self.X.requires_grad = True

        #optimizer settings
        self.adam = torch.optim.Adam(self.model.parameters())
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn='strong_wolfe'
        )

        self.criterion = torch.nn.MSELoss()
        self.iter = 1

    def loss_func(self):
        self.adam.zero_grad()
        self.optimizer.zero_grad()

        y_pre = self.model(self.X_train)
        loss_data = self.criterion(y_pre, self.y_train) #边界和初始条件

        u = self.model(self.X)
        du_dX = torch.autograd.grad(u, self.X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        du_dt = du_dX[:, 1]
        du_dx = du_dX[:, 0]

        #注意du_dXX的自变量和因变量
        du_dXX = \
        torch.autograd.grad(du_dx, self.X, grad_outputs=torch.ones_like(du_dx), create_graph=True, retain_graph=True)[0]

        du_dxx = du_dXX[:, 0]

        loss_pde = self.criterion(du_dt + u.squeeze() * du_dx, (0.01 / math.pi) * du_dxx) #方程残差
        loss = loss_data + loss_pde
        loss.backward()

        if self.iter % 100 == 0:
            print(f"Iteration: {self.iter}, loss: {loss.item()}")
        self.iter += 1
        return loss

    def train(self):
        self.model.train()
        for i in range(1000):
            self.adam.step(self.loss_func)
        self.optimizer.step(self.loss_func)

    def eval(self):
        self.model.eval()


net = Net()
net.train()

net.model.eval()

h = 0.01
k = 0.01
x = torch.arange(-1, 1 + h, h)
t = torch.arange(0, 1 + k, k)
X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T
X = X.to(net.X.device)
X
X.shape

model = net.model
model.eval()
with torch.no_grad():
    y_pre = model(X)
    y_pre = y_pre.reshape(len(x), len(t)).cpu().numpy()

plt.figure()
plt.plot(y_pre[:, 80])
plt.show()

"""


