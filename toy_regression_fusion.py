import sys
import numpy as np
from matplotlib import pyplot
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
    loss1 = torch.mul(torch.exp(-var),(pred - target) ** 2)
    loss2 = var
    loss = .5 * (loss1 + loss2)
    return loss.mean()







n_x_train = 1000   # the number of training datapoints
n_x_test = 100     # the number of testing datapoints

# x_train = np.random.rand(n_x_train,1)*18 - 9  # Initialize a vector of with dimensions [n_x, 1] and extend
# y_train = (np.cos(x_train))/2.5           # Calculate the sin of all data points in the x vector and reduce amplitude
# y_train += (np.random.randn(n_x_train, 1)/20)  # add noise to each datapoint

# x_test = np.random.rand(n_x_test, 1)*18 - 9   # Repeat data generation for test set
# y_test = (np.cos(x_test))/2.5
# y_test += (np.random.randn(n_x_test, 1)/20)


x_train_1 = np.expand_dims(np.linspace(-1.,-0.5, 200),1)
x_train_2 = np.expand_dims(np.linspace(-0.5, 0.,300),1)
x_train_3 = np.expand_dims(np.linspace(0., 0.5, 100), 1)
x_train_4 = np.expand_dims(np.linspace(0.5, 1., 400), 1)
x_train = np.concatenate([x_train_1, x_train_2, x_train_3, x_train_4])
y_train = np.cos(x_train)/2.5
y_train += (np.random.randn(n_x_train, 1)/22)

x_test = np.expand_dims(np.linspace(-1,1,n_x_test),1)
y_test = np.cos(x_test)/2.5
y_test += (np.random.randn(n_x_test, 1)/22)

print("x min: ", min(x_train))
print("x max:", max(x_train))
print("y min: ", min(y_train))
print("y max:", max(y_train))


class SineDataset(Dataset):
    def __init__(self, x, y):
        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor     # for MSE or L1 Loss

        self.length = x.shape[0]

        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def train_batch(model, x, y, optimizer, loss_fn):
    # Run forward calculation
    y_predict, variance = model.forward(x)

    # Compute loss.
    loss = loss_fn(y_predict, variance, y)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    return loss.data.item()


def train(model, loader, optimizer, loss_fn, epochs=5):
    losses = list()

    batch_index = 0
    for e in range(epochs):
        for x, y in loader:
            loss = train_batch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)

            batch_index += 1

        print("Epoch: ", e+1)
        print("Batches: ", batch_index)

    return losses


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

def plot_loss(losses, show=True):
    fig = pyplot.gcf()
    fig.set_size_inches(8,6)
    ax = pyplot.axes()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    x_loss = list(range(len(losses)))
    pyplot.plot(x_loss, losses)
    pyplot.savefig('loss.png')
    pyplot.close()

class ShallowLinear(nn.Module):
    '''
    A simple, general purpose, fully connected network
    '''
    def __init__(self, p):
        # Perform initialization of the pytorch superclass
        super(ShallowLinear, self).__init__()
        
        # Define network layer dimensions
        D_in, H1, H2, H3, D_out = [1, 64, 64, 64, 2]    # These numbers correspond to each layer: [input, hidden_1, output]
        
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
        # x = F.dropout(input = self.linear1(x), p = self.dropout_rate, training = True) # hidden layer
        # x = torch.relu(x)       # activation function
        
        # x = F.dropout(input = self.linear2(x), p = self.dropout_rate, training = True) # hidden layer
        # x = torch.relu(x)       # activation function
        
        # x = F.dropout(input = self.linear3(x), p = self.dropout_rate, training = True) # hidden layer
        # x = torch.relu(x)       # activation function

        x = self.dropout(self.linear1(x)) # hidden layer
        x = torch.relu(x)       # activation function
        
        x = self.dropout(self.linear2(x)) # hidden layer
        x = torch.relu(x)       # activation function
        
        x = self.dropout(self.linear3(x)) # hidden layer
        x = torch.relu(x)       # activation function

        out = self.linear4(x) # output layer
        
        return out[:,0].unsqueeze(1), out[:,1].unsqueeze(1)

def run(dataset_train, dataset_test):
    # Batch size is the number of training examples used to calculate each iteration's gradient
    batch_size_train = 16
    
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=False)
    
    # Define the hyperparameters
    learning_rate = 1e-3
    shallow_model = ShallowLinear(p = 0.5)
    
    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(shallow_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Define the loss function
    # loss_fn = nn.MSELoss()  # mean squared error

    # Train and get the resulting loss per iteration
    loss = train(model=shallow_model, loader=data_loader_train, optimizer=optimizer, loss_fn=heteroscedastic_loss)
    
    # Test and get the resulting predicted y values
    y_predict = test(model=shallow_model, loader=data_loader_test, mc_samples = 1000, epistemic = True)
    _, y_ale_variance = test(model=shallow_model, loader=data_loader_test, mc_samples = None, epistemic = False)

    return loss, y_predict, y_ale_variance


dataset_train = SineDataset(x=x_train, y=y_train)
dataset_test = SineDataset(x=x_test, y=y_test)

print("Train set size: ", dataset_train.length)
print("Test set size: ", dataset_test.length)

losses, y_predict, y_ale_variance = run(dataset_train=dataset_train, dataset_test=dataset_test)

# Mean and epistemic variance of predictions
y_predict_mean = np.mean(y_predict, axis = 0)
epistemic_uncertainty = np.mean(y_predict**2, axis = 0) - np.square(np.mean(y_predict, axis = 0)) 
# Aleatoric variance 
y_ale_variance = np.exp(-y_ale_variance)

print("Final loss:", sum(losses[-100:])/100)
plot_loss(losses)

fig2 = pyplot.figure()
fig2.set_size_inches(8,6)
pyplot.scatter(x_train, y_train, marker='o', color = 'red', s=0.2)  # plot x vs y
# pyplot.scatter(x_test, y_test, marker='o', s=0.2)
pyplot.scatter(x_test, y_test, marker='o', color = 'blue', s=1)
pyplot.scatter(x_test, y_predict_mean, marker='o', color = 'yellow', s=0.1)
pyplot.fill_between(x_test.ravel(), (y_predict_mean - y_ale_variance).ravel(), (y_predict_mean + y_ale_variance).ravel(), alpha = 0.5)
# pyplot.text(-9, 0.40, "- Training data", color="yellow", fontsize=8)
# pyplot.text(-9, 0.44, "- Prediction", color="orange", fontsize=8)
# pyplot.text(-9, 0.48, "- Sine (with noise)", color="blue", fontsize=8)
pyplot.savefig('ale_100Train_1.jpg')
pyplot.close()