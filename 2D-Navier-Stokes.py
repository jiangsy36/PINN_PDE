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

import deepxde as dde
from deepxde.backend import tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许多个 OpenMP 运行时

torch.manual_seed(42)
np.random.seed(42)


rho = 1
mu = 1
u_in = 1
D = 1
L = 1

geom = dde.geometry.Rectangle(xmin=[-L/2,-D/2], xmax=[L/2,D/2])

def boundary_wall(X, on_boundary):
    print("X", X)
    print("on_boundary", on_boundary)
    on_wall = np.logical_and(np.logical_or(np.isclose(X[1],-D/2,rtol=1e-5,atol=1e-8),np.isclose(X[1],D/2,rtol=1e-5,atol=1e-8)),on_boundary)

    return on_wall

def boundary_inlet(X, on_boundary):
    on_inlet = np.logical_and(np.isclose(X[0],-L/2,rtol=1e-5,atol=1e-8),on_boundary)
    return on_inlet

def boundary_outlet(X, on_boundary):
    on_outlet = np.logical_and(np.isclose(X[0],L/2,rtol=1e-5,atol=1e-8),on_boundary)
    return on_outlet

bc_wall_u = dde.DirichletBC(geom, lambda X:0., boundary_wall, component=0)
bc_wall_v = dde.DirichletBC(geom, lambda X:0., boundary_wall, component=1)
bc_inlet_u = dde.DirichletBC(geom,  lambda X: u_in, boundary_inlet, component=0)
bc_inlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_inlet, component=1)
bc_outlet_p = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component=2)
bc_outlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component=1)

def pde(X,Y):
    du_x = dde.grad.jacobian(Y,X,i=0,j=0)
    du_y = dde.grad.jacobian(Y,X,i=0,j=1)
    dv_x = dde.grad.jacobian(Y,X,i=1,j=0)
    dv_y = dde.grad.jacobian(Y,X,i=1,j=1)

    dp_x = dde.grad.jacobian(Y,X,i=2,j=0)
    dp_y = dde.grad.jacobian(Y,X,i=2,j=1)

    du_xx = dde.grad.hessian(Y,X,component=0,i=0,j=0)
    du_yy = dde.grad.hessian(Y,X,component=0,i=1,j=1)
    dv_xx = dde.grad.hessian(Y,X,component=1,i=0,j=0)
    dv_yy = dde.grad.hessian(Y,X,component=1,i=1,j=1)

    pde_u = Y[:,0:1]*du_x + Y[:,1:2]*du_y +1/rho*dp_x -mu*(du_xx+du_yy)
    pde_v = Y[:,0:1]*dv_x + Y[:,1:2]*dv_y +1/rho*dp_y -mu*(dv_xx+dv_yy)

    return [pde_u,pde_v]

data = dde.data.PDE(
    geom,
    pde,
    [bc_wall_u,bc_wall_v,bc_inlet_u,bc_inlet_v,bc_outlet_p,bc_outlet_v],
    num_domain=2000,
    num_boundary=200,
    num_test=200)
"""
plt.figure()
plt.scatter(data.train_x_all[:,0],data.train_x_all[:,1],s=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""
net = dde.maps.FNN([2]+[64]*3+[3],"tanh","Glorot normal")

model = dde.Model(data, net)
model.compile('adam',lr=0.001)

losshistory, train_state = model.train(epochs=10000)

dde.optimizers.config.set_LBFGS_options(maxcor=3000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

sample = geom.random_points(5000000)
result = model.predict(sample)

color_legend = [[0,1.5],[-0.3,0.3],[0.35]]

for idx in range(3):
    plt.figure()
    plt.scatter(sample[:,0],sample[:,1],c=result[:,idx], cmap = "rainbow", s=2)
    plt.colorbar()
    plt.clim(color_legend[idx])
    plt.xlim([-L/2,L/2])
    plt.ylim([-D/2,D/2])
    plt.tight_layout()
    plt.show()
