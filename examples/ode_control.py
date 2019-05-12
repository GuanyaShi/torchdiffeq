import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()


if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
t = torch.linspace(0.0, 0.1, args.batch_time)


# True dynamics (pendulum)
class Lambda(nn.Module):

    def __init__(self, u):
        super(Lambda, self).__init__()

        self.u = u

    def forward(self, t, y):
        return torch.tensor([[y[0, 1], torch.sin(y[0, 0]) + self.u]])


def get_true_y(u):
    true_y0 = torch.rand(1, 2) - 0.5
    
    with torch.no_grad():
        true_y = odeint(Lambda(u = u), true_y0, t, method='dopri5')

    #print(true_y0)
    #print(u)
    #print(true_y)

    return true_y0, true_y


def get_batch():    
    u = torch.rand(1) - 0.5

    batch_y0 = torch.zeros((args.batch_size, 1, 2)) # (M, D)
    batch_y = torch.zeros((args.batch_time, args.batch_size, 1, 2)) # (T, M, D)

    for i in range(args.batch_size):
        true_y0, true_y = get_true_y(u)
        batch_y0[i, :, :] = true_y0
        batch_y[:, i, :, :] = true_y

    batch_t = t[:args.batch_time]  # (T) 

    #print(batch_t)
    #print(u)
    #print(batch_y0)
    #print(batch_y)

    return batch_y0, batch_t, batch_y, u


class ODEFunc(nn.Module):

    def __init__(self, u):
        super(ODEFunc, self).__init__()
        self.u = u

        self.net = nn.Sequential(
            nn.Linear(3, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 2)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        U = self.u * torch.ones((args.batch_size, 1, 1))
        combined = torch.cat((y, U), dim=2)
        #print(combined)
        return self.net(combined)


# For testing
T = torch.linspace(0.0, 2.0, 200)

test_y0 = torch.zeros((args.batch_size, 1, 2)) # (M, D)
test_y = torch.zeros((200, args.batch_size, 1, 2)) # (T, M, D)

test_u = torch.rand(1) - 0.5
for i in range(args.batch_size):
    true_y0 = torch.rand(1, 2) - 0.5
    
    with torch.no_grad():
        true_y = odeint(Lambda(u = test_u), true_y0, T, method='dopri5')

    test_y0[i, :, :] = true_y0
    test_y[:, i, :, :] = true_y


# Main
if __name__ == '__main__':

    func = ODEFunc(u = 0.0)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y, u = get_batch()
        func.u = u
        pred_y = odeint(func, batch_y0, batch_t)

        #print('******************')

        #print(func.u)
        #print(batch_y0)
        #print(batch_t)
        #print(batch_y)

        #print('*** Prediction ***')
        #print(pred_y)

        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        if itr % args.test_freq == 0:
            print('u = ' + repr(u))
            with torch.no_grad():
                func.u = test_u
                Pred_y = odeint(func, test_y0, T)
                loss = torch.mean(torch.abs(Pred_y - test_y))
            print(loss.item())
        #     with torch.no_grad():
        #         pred_y = odeint(func, true_y0, t)
        #         loss = torch.mean(torch.abs(pred_y - true_y))
        #         print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
        #         ii += 1