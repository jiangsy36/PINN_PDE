#m*d^2x(t)/dt^2 + c*dx(t)/dt + k*x(t) = F(t)

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

m = 1.0 #mass
c = 0.1 #damping
k = 2.0 #spring constant

def F(t):
    return 0

#initial conditions
v0 = 1.0 #dx/dt initial velocity
x0 = 1.0 #x initial displacement

#time domain
t_min = 0.0
t_max = 10.0

#number of collocation points
N_f = 10000 #for the ODE residual

#collocation points in the interior of domain
t_f  = torch.linspace(t_min, t_max, N_f).reshape(-1, 1)
#print(t_f.shape)

t_f.requires_grad = True #Enable gradiant computation on t_f

#initial condition point
t_i = torch.tensor([[t_min]],dtype=torch.float32,requires_grad=True)
#print(t_i)

#Define the neural network model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        number_hidden_layers = 4
        number_neurons_per_layer = 20

        #input layer
        self.input_layer = nn.Linear(1, number_neurons_per_layer)
        self.hidden_layers.append(nn.Tanh())

        #hidden layers
        for _ in range(number_hidden_layers):
            layer = nn.Linear(number_neurons_per_layer, number_neurons_per_layer)
            init.xavier_normal_(layer.weight) #Xavier initialization
            self.hidden_layers.append(layer)
            self.hidden_layers.append(nn.Tanh())

        #output layer
        self.output_layer = nn.Linear(number_neurons_per_layer, 1)
        init.xavier_normal_(self.output_layer.weight) #Xavier initialization

    def forward(self, t):
        X = torch.tanh(self.input_layer(t))
        for layer in self.hidden_layers:
            X = layer(X)

        return self.output_layer(X)

#instantiate the model
model = PINN()

#Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

def loss_fn(model, t_f, t_i, x0, v0):
    x = model(t_f)
    x_t = torch.autograd.grad(x, t_f, torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t_f, torch.ones_like(x_t), create_graph=True)[0]

    f_ode = m*x_tt + c*x_t + k*x - F(t_f)
    f_ode_loss = torch.mean(f_ode**2)

    x_i = model(t_i)
    x_t_i = torch.autograd.grad(x_i, t_i, torch.ones_like(x_i), create_graph=True)[0]
    ic_loss = (x_i - x0)**2 + (x_t_i - v0)**2

    total_loss = f_ode_loss + ic_loss

    return total_loss

#Training
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    loss_value = loss_fn(model, t_f, t_i, torch.tensor([[x0]], dtype=torch.float32), torch.tensor([[v0]], dtype=torch.float32))

    loss_value.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print('Epoch: {}, Loss: {}'.format(epoch, loss_value.item()))

#Calculate the analytical solution
omega_n = np.sqrt(k/m)
zeta = c/(2*np.sqrt(k*m))

if zeta < 1:
    #under-damped case
    omega_d  = omega_n*np.sqrt(1-zeta**2)
    A = x0
    B = (v0+zeta*omega_n*x0)/omega_d
    t_analytical = np.linspace(t_min, t_max, 1000)
    x_analytical = np.exp(-zeta*omega_n*t_analytical)*(A*np.cos(omega_d*t_analytical)+B*np.sin(omega_d*t_analytical))
else:
    #critically damped or over-damped case
    t_analytical = np.linspace(t_min, t_max, 1000)
    x_analytical = np.zeros_like(t_analytical)

#predict displacement using the trained model
x_test = torch.tensor(t_analytical, dtype=torch.float32).reshape(-1, 1)
x_pinn = model(x_test).detach().numpy().flatten()

plt.figure()
plt.plot(t_analytical, x_analytical, label='analytical', linestyle='--')
plt.plot(t_analytical, x_pinn, label='PINN', linestyle='solid')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Mass-Spring-Damper System')
plt.legend()
plt.grid(True)
plt.show()