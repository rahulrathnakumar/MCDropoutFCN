import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from sklearn.datasets import make_moons


class NNLinear(nn.Module):
    def __init__(self, p):
        # Perform initialization of the pytorch superclass
        super(NNLinear, self).__init__()
        
        # Define network layer dimensions
        D_in, H1, H2, H3, D_out = [2, 64, 64, 64, 4]    # These numbers correspond to each layer: [input, hidden_1, output]
        
        # Define layer types
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)
        self.dropout_rate = p
        self.dropout = nn.Dropout(p = p)
    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        x = self.dropout(self.linear1(x)) # hidden layer
        x = torch.relu(x)       # activation function
        
        x = self.dropout(self.linear2(x)) # hidden layer
        x = torch.relu(x)       # activation function
        
        x = self.dropout(self.linear3(x)) # hidden layer
        x = torch.relu(x)       # activation function

        out = self.linear4(x) # output layer
        
        return out

class Gaussian():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MoonsDataset(Dataset):
    def __init__(self, x, y):
        x_dtype = torch.FloatTensor
        y_dtype = torch.LongTensor     # NLL Loss
        self.length = x.shape[0]
        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.length


def plot_loss(losses, show=True):
    fig = plt.gcf()
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    x_loss = list(range(len(losses)))
    plt.plot(x_loss, losses)
    plt.savefig('loss.png')
    plt.close()



def stochastic_log_softmax(mean, var):
    '''
    Returns Lx in the paper
    '''
    # Sample 't' times from a normal distribution to get x_hat
    x_hats = list()
    for t in range(10):
        eps = torch.normal(mean = torch.tensor(0.), std = torch.tensor(1.))
        x_hat = mean + var*eps
        x_hats.append(x_hat)
    x_hats = torch.stack(x_hats)
    return torch.mean(torch.log_softmax(x_hats , dim = 2), dim = 0)

def train_batch(model, x, y, optimizer, loss_fn):
    out = model.forward(x)
    y_predict = out[:,:2]
    var = out[:,2:]
    # Compute stochastic NLL Loss
    Lx = stochastic_log_softmax(y_predict, var)
    loss = loss_fn(Lx, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.data.item()

def train(model, loader, optimizer, loss_fn, epochs=200):
    losses = list()

    batch_index = 0
    for e in range(epochs):
        for x, y in loader:
            loss = train_batch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)

            batch_index += 1

        print("Epoch: ", e+1)
        print("loss: ", loss)
        print("Batches: ", batch_index)

    return losses


def test_batch(model, x, y):
    # run forward calculation
    out = model.forward(x)
    y_predict = out[:,:2]
    y_ale_variance = out[:,2:]
    return y, y_predict, y_ale_variance

def test(model, loader, mc_samples, epistemic, decision_boundary = False):
    y_vectors = list()
    y_predict_vectors = list()
    y_predict_ale_vectors = list()
    y_predicts = list()
    batch_index = 0
    model.eval()
    if epistemic == True:
        model.dropout.train()
        for i in range(mc_samples):
            for x, y in loader:
                y, y_predict, _ = test_batch(model=model, x=x, y=y)
                y_predict = torch.nn.functional.softmax(y_predict)
                y_vectors.append(y.data)
                y_predict_vectors.append(y_predict.data)
                batch_index += 1
        y_predicts = torch.stack(y_predict_vectors)
        return y_predicts
    else:
        for x,y in loader:
            y, y_predict, y_ale_variance = test_batch(model=model, x=x, y=y)
            y_predict = torch.nn.functional.softmax(y_predict)
            y_vectors.append(y.data)
            y_predict_vectors.append(y_predict.data)
            y_predict_ale_vectors.append(y_ale_variance.data)
            batch_index += 1
        return torch.stack(y_predict_vectors), torch.stack(y_predict_ale_vectors)[0]


def plot_uncertain_decision_boundary(model, mc_samples):
    x_min, x_max = x_train[:, 0].min()-0.1,x_train[:, 0].max()+0.1
    y_min, y_max = x_train[:, 1].min()-0.1, x_train[:, 1].max()+0.1

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 200

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                np.arange(y_min, y_max, spacing))

    # Concatenate data to match input
    data = np.hstack((XX.ravel().reshape(-1,1), 
                    YY.ravel().reshape(-1,1)))

    outs = list()
    data_t = torch.FloatTensor(data)
    model.eval()
    model.dropout.train()
    for i in range(mc_samples):
        out = model(data_t)
        clf = np.where(torch.nn.functional.softmax(out[:,:2])[:,1]<0.5,0,1)
        Z = clf.reshape(XX.shape)
        outs.append(Z)
    fig = plt.figure()
    for o in outs:
        plt.contourf(XX, YY, o, cmap=plt.cm.Accent, alpha=0.5)
    plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
    plt.savefig('100Train.jpg')
    plt.close()


def run(dataset_train, dataset_test):
    batch_size_train = 16
    
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=False)
    
    # Define the hyperparameters
    learning_rate = 1e-3
    net = NNLinear(p = 0.1)
    
    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = nn.NLLLoss()
    # Train and get the resulting loss per iteration
    loss = train(model=net, loader=data_loader_train, optimizer=optimizer, loss_fn=loss_fn)
    
    # Test and get the resulting predicted y values
    y_predict = test(model=net, loader=data_loader_test, mc_samples = 1000, epistemic = True)
    _, y_ale_variance = test(model=net, loader=data_loader_test, mc_samples = 1000, epistemic = False)
    plot_uncertain_decision_boundary(model = net, mc_samples = 100)
    return loss, y_predict, y_ale_variance



n_x_train = 100   # the number of training datapoints
n_x_test = 100     # the number of testing datapoints

# Construct dataset
x_train, y_train = make_moons(n_samples=int(n_x_train), noise = 0.1)

x_test, y_test = make_moons(n_samples=int(n_x_test), noise = 0.1)

dataset_train = MoonsDataset(x=x_train, y=y_train)
dataset_test = MoonsDataset(x=x_test, y=y_test)

print("Train set size: ", dataset_train.length)
print("Test set size: ", dataset_test.length)

losses, y_predict, y_ale_variance = run(dataset_train=dataset_train, dataset_test=dataset_test)

# Mean and epistemic variance of predictions
y_predict_mean = torch.mean(y_predict, dim = 0)
epistemic_uncertainty = torch.mean(y_predict, dim = 0) - (torch.mean(y_predict, dim = 0))**2
# Aleatoric variance 
y_ale_variance = np.exp(y_ale_variance)

print("Final loss:", sum(losses[-100:])/100)


