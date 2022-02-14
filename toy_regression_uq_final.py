from re import M
import sys
import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F


def heteroscedastic_loss(pred, var, target):
    """
    MSE loss dependent on data variance
    """
    loss1 = (1/torch.exp(var))*(pred - target)**2 # Assuming this is hadamard...
    loss2 = var
    print(var)
    loss = .5 * (loss1 + loss2)
    return loss.mean(), loss1.mean(), loss2.mean()


n_x_train = 100   # the number of training datapoints
n_x_test = 100     # the number of testing datapoints

# x_train = np.random.rand(n_x_train,1)*18 - 9  # Initialize a vector of with dimensions [n_x, 1] and extend
# y_train = (np.cos(x_train))/2.5           # Calculate the sin of all data points in the x vector and reduce amplitude
# y_train += (np.random.randn(n_x_train, 1)/20)  # add noise to each datapoint

# x_test = np.random.rand(n_x_test, 1)*18 - 9   # Repeat data generation for test set
# y_test = (np.cos(x_test))/2.5
# y_test += (np.random.randn(n_x_test, 1)/20)


# x_train_1 = np.expand_dims(np.linspace(-1.0,0, 50),1)
# x_train_2 = np.expand_dims(np.linspace(0.0, 1.0,50),1)
# x_train_3 = np.expand_dims(np.linspace(0.1, 1., 2), 1)
# x_train_4 = np.expand_dims(np.linspace(0.9, 1., 50), 1)
# x_train = np.concatenate([x_train_1, x_train_2])
x_train = np.expand_dims(np.linspace(-1,1,n_x_train),1)
# y_train_1 = np.cos(x_train_1)/10
# y_train_1 += (np.random.randn(50, 1)/200)
# y_train_2 = np.cos(x_train_2)/10
# y_train_2 += (np.random.randn(50, 1)/20)
# y_train = np.concatenate([y_train_1, y_train_2])
y_train = np.cos(x_train)/2.5
y_train += np.random.randn(n_x_train, 1)/50


x_test = np.expand_dims(np.linspace(-1,1,n_x_test),1)
y_test = np.cos(x_test)/2.5
y_test += (np.random.randn(n_x_test, 1)/50)

print("x min: ", min(x_train))
print("x max:", max(x_train))
print("y min: ", min(y_train))
print("y max:", max(y_train))


class SineDataset(Dataset):
    def __init__(self, x, y):
        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor     

        self.length = x.shape[0]

        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def train_batch(model, x, y, optimizer, loss_fn):
    y_predict, variance = model.forward(x)

    # Compute loss.
    loss, loss1, loss2 = loss_fn(y_predict, variance, y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.data.item(), loss1.data.item(), loss2.data.item()


def train(model, loader, optimizer, loss_fn, epochs=5):
    losses = list()
    losses_1 = list()
    losses_2 = list()
    batch_index = 0
    for e in range(epochs):
        for x, y in loader:
            loss, loss1, loss2 = train_batch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)
            losses_1.append(loss1)
            losses_2.append(loss2)
            batch_index += 1

        print("Epoch: ", e+1)
        print("Batches: ", batch_index)

    return losses, losses_1, losses_2


def test_batch(model, x, y):
    # run forward calculation
    y_predict, y_ale_variance = model.forward(x)

    return y, y_predict, y_ale_variance


def test(model, loader, mc_samples, epistemic):
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
                y_vectors.append(y.data.numpy())
                y_predict_vectors.append(y_predict.data.numpy())

                batch_index += 1
        y_predicts = np.stack(y_predict_vectors)
        return y_predicts
    else:
        for x,y in loader:
            y, y_predict, y_ale_variance = test_batch(model=model, x=x, y=y)
            y_vectors.append(y.data.numpy())
            y_predict_vectors.append(y_predict.data.numpy())
            y_predict_ale_vectors.append(y_ale_variance.data.numpy())

            batch_index += 1
        return np.stack(y_predict_vectors), np.stack(y_predict_ale_vectors)[0]

def plot_loss(losses, figname, show=True):
    fig = pyplot.gcf()
    fig.set_size_inches(8,6)
    ax = pyplot.axes()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    x_loss = list(range(len(losses)))
    pyplot.plot(x_loss, losses)
    pyplot.savefig(figname)
    pyplot.close()

class ShallowLinear(nn.Module):
    '''
    A simple, general purpose, fully connected network
    '''
    def __init__(self, p):
        # Perform initialization of the pytorch superclass
        super(ShallowLinear, self).__init__()
        
        # Define network layer dimensions
        D_in, H1, H2, H3, D_out = [1, 256, 512, 256, 2]    # These numbers correspond to each layer: [input, hidden_1, output]
        
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
        return out[:,0].unsqueeze(1), out[:,1].unsqueeze(1)

def run(dataset_train, dataset_test, mc_samples):
    # Batch size is the number of training examples used to calculate each iteration's gradient
    batch_size_train = 20
    
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=False)
    
    # Define the hyperparameters
    learning_rate = 1e-3
    shallow_model = ShallowLinear(p = 0.1)
    
    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(shallow_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Define the loss function
    # loss_fn = nn.MSELoss()  # mean squared error

    # Train and get the resulting loss per iteration
    loss, loss1, loss2 = train(model=shallow_model, loader=data_loader_train, optimizer=optimizer, loss_fn=heteroscedastic_loss, epochs=100)
    
    # Test and get the resulting predicted y values
    y_predict = test(model=shallow_model, loader=data_loader_test, mc_samples = mc_samples, epistemic = True)
    _, y_ale_variance = test(model=shallow_model, loader=data_loader_test, mc_samples = None, epistemic = False)

    return loss, loss1, loss2, y_predict, y_ale_variance


dataset_train = SineDataset(x=x_train, y=y_train)
dataset_test = SineDataset(x=x_test, y=y_test)

print("Train set size: ", dataset_train.length)
print("Test set size: ", dataset_test.length)
mc_samples = 1000
losses, losses_1, losses_2, y_predict, y_ale_variance = run(dataset_train=dataset_train, dataset_test=dataset_test, mc_samples= mc_samples)

# Mean and epistemic variance of predictions
y_predict_mean = np.mean(y_predict, axis = 0)
epistemic_uncertainty = np.mean(y_predict**2, axis = 0) - np.square(np.mean(y_predict, axis = 0)) 
# Aleatoric variance 
aleatoric_uncertainty = np.exp(-y_ale_variance)

# Scaling the uncertainties
def minmaxscaling(x, min, max):
    x_std = (x - x.min())/(x.max() - x.min())
    x_scaled = x_std*(max - min ) + min
    return x_scaled

scaled_epistemic_uncertainty = minmaxscaling(epistemic_uncertainty, min = 0, max = 1)
scaled_aleatoric_uncertainty = minmaxscaling(aleatoric_uncertainty, min = 0, max = 1)


print("Final loss:", sum(losses[-100:])/100)
plot_loss(losses, figname= 'loss_total.png', show = False)
plot_loss(losses_1, figname = 'loss_component_main.png',  show = False)
plot_loss(losses_2, figname = 'loss_residual.png',  show = False)


fig2 = pyplot.figure()
fig2.set_size_inches(8,6)
# pyplot.scatter(x_train, y_train, marker='o', color = 'green', s=1.0)  # plot x vs y
# pyplot.scatter(x_test, y_test, marker='o', color = 'yellow', s=1.0)
pyplot.scatter(x_test, y_predict_mean, color = 'red', s=1.0)
pyplot.fill_between(x_test.ravel(), (y_predict_mean - 50*epistemic_uncertainty).ravel(), (y_predict_mean + 50*epistemic_uncertainty).ravel(), alpha = 0.5)
# pyplot.text(-9, 0.40, "- Training data", color="yellow", fontsize=8)
# pyplot.text(-9, 0.44, "- Prediction", color="orange", fontsize=8)
# pyplot.text(-9, 0.48, "- Sine (with noise)", color="blue", fontsize=8)
pyplot.savefig('epistemic_uncertainty_10000.jpg')
pyplot.close()

fig2 = pyplot.figure()
fig2.set_size_inches(8,6)
# pyplot.scatter(x_train, y_train, marker='o', color = 'green', s=1.0)  # plot x vs y
# pyplot.scatter(x_test, y_test, marker='o', color = 'yellow', s=1.0)
pyplot.scatter(x_test, y_predict_mean, color = 'red', s=1.0)
pyplot.fill_between(x_test.ravel(), (y_predict_mean - y_ale_variance/100).ravel(), (y_predict_mean + y_ale_variance/100).ravel(), alpha = 0.5)
# pyplot.text(-9, 0.40, "- Training data", color="yellow", fontsize=8)
# pyplot.text(-9, 0.44, "- Prediction", color="orange", fontsize=8)
# pyplot.text(-9, 0.48, "- Sine (with noise)", color="blue", fontsize=8)
pyplot.savefig('aleatoric_uncertainty_10000.jpg')
pyplot.close()