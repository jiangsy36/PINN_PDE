

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


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def initial_condition(x, y):
    return torch.sin(torch.pi*x)*torch.sin(torch.pi*y)

def boundary_condition(x, y, t, custom_value):
    return torch.full_like(x, custom_value)
"""
# Example usage
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = None  # not used
t = None  # not used
custom_value = 5

result = boundary_condition(x, y, t, custom_value)
print(result)  # Outputs a tensor filled with the value 5.
"""

def generate_training_data(n_points):
    x = torch.rand(n_points, 1, requires_grad=True)
    y = torch.rand(n_points, 1, requires_grad=True)
    t = torch.rand(n_points, 1, requires_grad=True)

    return x, y, t

def generate_boundary_points(num_points):
    x_boundary = torch.tensor([0.0,1.0]).repeat(num_points//2)
    y_boundary = torch.rand(num_points)

    if torch.rand(1) > 0.5:
        x_boundary, y_boundary = y_boundary, x_boundary

    return x_boundary.view(-1,1), y_boundary.view(-1,1)

def generate_boundary_training_data(num_points):
    x_boundary, y_boundary = generate_boundary_points(num_points)
    t = torch.rand(num_points, 1, requires_grad=True)

    return x_boundary, y_boundary, t

def pde(x, y, t, model):
    input_data = torch.cat((x,y,t),dim=1)
    u = model(input_data)
    u_x, u_y = torch.autograd.grad(u, [x,y], grad_outputs=torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    heat_eq_residual = 1*u_xx - 1*u_yy-1*u_t
    return heat_eq_residual

def train_PINN(model,num_iterations,num_points):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        x, y, t = generate_training_data(num_points)
        x_b, y_b, t_b = generate_boundary_training_data(num_points)

        t_initial = torch.zeros_like(t)
        u_initial = initial_condition(x, y)

        custom_value = 0

        u_boundary_x = boundary_condition(x_b, y_b, t_b, custom_value)
        u_boundary_y = boundary_condition(y_b, x_b, t_b, custom_value)

        residual = pde(x, y, t, model)

        loss = nn.MSELoss()(u_initial,model(torch.cat([x,y,t_initial],dim=1))) +\
            nn.MSELoss()(u_boundary_x,model(torch.cat([x_b,y_b,t_b],dim=1)))+\
            nn.MSELoss()(u_boundary_y,model(torch.cat([y_b,x_b,t_b],dim=1)))+\
            nn.MSELoss()(residual,torch.zeros_like(residual))

        loss.backward()
        optimizer.step()

        if iteration % 100 ==0:
            print(f"Iteration: {iteration}, Loss: {loss.item()}")

model = PINN()
num_iterations = 10000
num_points = 1000
train_PINN(model,num_iterations,num_points)

with torch.no_grad():
    x_vals = torch.linspace(0, 1, 100)
    y_vals = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid([x_vals, y_vals])
    t_val = torch.ones_like(X)*0

    input_data = torch.stack([X.flatten(), Y.flatten(),t_val.flatten()],dim=1)
    solution = model(input_data).reshape(X.shape, Y.shape)

    plt.figure()
    sns.heatmap(solution,cmap='jet')
    plt.title("Solution to Heat Equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

with torch.no_grad():
    x_vals = torch.linspace(0, 1, 100)
    y_vals = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid([x_vals, y_vals])
    t_val = torch.ones_like(X)*10

    input_data = torch.stack([X.flatten(), Y.flatten(),t_val.flatten()],dim=1)
    solution = model(input_data).reshape(X.shape, Y.shape)

    plt.figure()
    sns.heatmap(solution,cmap='jet')
    plt.title("Solution to Heat Equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()