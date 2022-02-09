import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import numpy as np
# import imageio
import random

# def shuffle(x,y):
#     x = x[torch.randperm(len(x))]
#     y = y[torch.randperm(y.size()[0])]
#     return x, y


# def batch(x, y, batch_size):
#     x = torch.stack(torch.split(x, split_size_or_sections = batch_size))
#     y = torch.stack(torch.split(y, split_size_or_sections = batch_size))
#     return x, y

def heteroscedastic_loss(input, target):
    """
    MSE loss dependent on data variance
    """
    return ((F.mse_loss(input, target, reduction='none')/torch.var(input)) + (0.5*torch.log(torch.var(input)))).mean(dim = 0)


torch.manual_seed(1)    # reproducible
batch_size = 20
x = torch.unsqueeze(torch.linspace(-1, 1, 5000), dim=1)  # x data (tensor), shape=(100, 1)
y = torch.cos(x) + 0.1*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)



# torch can only train on Variable, so convert them to Variable
x, y = torch.tensor(x), torch.tensor(y)

x_val = torch.unsqueeze(torch.linspace(-1.,1., 1000), dim=1)  # x data (tensor), shape=(100, 1)
y_val = torch.cos(x_val) + 0.1*torch.rand(x_val.size())                 # noisy y data (tensor), shape=(100, 1)
x_val, y_val = torch.tensor(x_val), torch.tensor(y_val)


# x, y = batch(x, y, batch_size)
# x_val, y_val = batch(x_val, y_val, batch_size)

# this is one way to define a network
class Net(nn.Module):
    '''
    A simple, general purpose, fully connected network
    '''
    def __init__(self, p):
        # Perform initialization of the pytorch superclass
        super(Net, self).__init__()
        
        # Define network layer dimensions
        D_in, H1, H2, H3, D_out = [1, 64, 64, 64, 1]    # These numbers correspond to each layer: [input, hidden_1, output]
        
        # Define layer types
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)
        self.dropout = nn.Dropout(p = p)

    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        x = self.linear1(x) # hidden layer
        x = torch.tanh(x)       # activation function
        
        x = self.linear2(x) # hidden layer
        x = torch.tanh(x)       # activation function
        
        x = self.linear3(x) # hidden layer
        x = torch.tanh(x)       # activation function

        x = self.linear4(x) # output layer
        
        return x
net = Net(p = 0.1)     # define the network
net.train()
# print(net)  # net architecture
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

my_images = []
fig, ax = plt.subplots(figsize=(12,7))
best_loss = 1E10
# train the network
for t in range(1000):
    # x, y = shuffle(x,y)
    # for i, batch in enumerate(zip(x,y)):
    prediction = net(x[torch.randperm(x.size()[0])])     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    print("Loss: ", loss)
    if loss < best_loss:
        best_loss = loss
        checkpoint = {
            'epoch': t + 1,
            'net_state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }

torch.save(checkpoint, 'models/verification/checkpoint.pt')
checkpoint = torch.load('models/verification/checkpoint.pt')
net.load_state_dict(checkpoint['net_state_dict'])
# Validate on val_set
net.eval()
print("Validating at Epoch: ", checkpoint['epoch'])
# Epistemic uncertainty
net.dropout.train()
num_samples = 500

with torch.no_grad():
    outs = []
    for i in range(num_samples):
        out = net(x_val)
        outs.append(out)

# Compute variance of outputs
outs = torch.stack(outs)
mean_ = torch.mean(outs, dim = 0)
std_ = torch.std(outs, dim = 0)
# view data
plt.figure(figsize=(10,4))
plt.scatter(x.data.numpy(), y.data.numpy(), color = "black", s = 2)
# plt.scatter(x_val.data.numpy(), outs[0].data.numpy(), color = "red")
# plt.scatter(x_val.data.numpy(), outs[1].data.numpy(), color = "green")
# plt.scatter(x_val.data.numpy(), outs[2].data.numpy(), color = "blue")
plt.scatter(x_val.data.numpy(), mean_.data.numpy(), color = "blue", s = 2)
plt.scatter(x_val.data.numpy(), y_val.data.numpy(), color = "yellow", s = 2)
plt.fill_between(x_val.data.numpy().ravel(), (mean_ - std_).data.numpy().ravel(), (mean_ + std_).data.numpy().ravel(), alpha = 0.8)
plt.title('Regression Analysis')
plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')
# plt.savefig('regression_uq_withinSet_dropout0.1_1000Train.jpg')
plt.savefig('cosineCurve_1000Train_3Layer.jpg')
# Epistemic uncertainty - Continued...
