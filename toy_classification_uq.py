import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

import numpy as np
# import imageio


torch.manual_seed(1)    # reproducible


# Epistemic Uncertainty
def entropy(preds):
    N,C = preds.shape
    entropy_c = []
    for c in range(C):
        entropy_c.append(preds[:,c]*np.log2(preds[:,c] + 1e-10))
    uncertainty = -np.sum(np.asarray(entropy_c), axis = 0)
    # uncertainty = np.asarray(entropy_c)[1]
    return uncertainty

num_training = np.linspace(100,1000, num = 10)
mc_samples = 1000
p = 0.25
for n_train_samples in num_training:
    # Dataset
    # Construct dataset
    x, y = make_gaussian_quantiles(cov=3.,
                                    n_samples=int(n_train_samples), n_features=2,
                                    n_classes=2, random_state=1)

    x_val, y_val = make_gaussian_quantiles(cov=3.,
                                    n_samples=100, n_features=2,
                                    n_classes=2, random_state=1)

    # torch can only train on Variable, so convert them to Variable
    x, y = torch.tensor(x).float(), torch.nn.functional.one_hot(torch.tensor(y), num_classes = 2).float()
    x_val, y_val = torch.tensor(x_val).float(), torch.nn.functional.one_hot(torch.tensor(y_val), num_classes = 2).float()


    # this is one way to define a network
    class Net(torch.nn.Module):
        def __init__(self, n_features, n_hidden, n_output, p):
            super(Net, self).__init__()
            self.hidden_1 = torch.nn.Linear(n_features, n_hidden)   # hidden layer1
            self.hidden_2 = torch.nn.Linear(n_hidden, 2*n_hidden)
            self.hidden_3 = torch.nn.Linear(2*n_hidden, 2*n_hidden)
            self.predict = torch.nn.Linear(2*n_hidden, n_output)   # output layer
            self.dropout = torch.nn.Dropout(p = p)
        def forward(self, x):
            x = self.dropout(F.relu(self.hidden_1(x)))      # activation function for hidden layer
            x = self.dropout(F.relu(self.hidden_2(x)))
            x = self.dropout(F.relu(self.hidden_3(x))) 
            x = self.predict(x)             # linear output
            return x

    net = Net(n_features=2, n_hidden=10, n_output=2, p = p)     # define the network
    net.train()
    # print(net)  # net architecture
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = torch.nn.BCEWithLogitsLoss()


    # train the network
    for t in range(500):
    
        prediction = net(x)     # input x and predict based on x

        loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        print("Loss: ", loss)
        checkpoint = {
            'epoch': t + 1,
            'net_state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
    # Torch - save model
    torch.save(checkpoint, 'models/verification_classifier/checkpoint.pt')

    def apply_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()

    net.eval()

    net.apply(apply_dropout)

    # Epistemic uncertainty

    with torch.no_grad():
        outs = []
        for i in range(mc_samples):
            out = net(x_val)
            # out_normalized = (out - torch.min(out, dim = 0)[0])/(torch.max(out, dim = 0)[0] - torch.min(out, dim = 0)[0])
            outs.append(out)
    outs = np.asarray(np.stack(outs))    
    outs_mean = np.mean(outs, axis = 0)
    # Normalize to 0-1

    # outs_mean = (outs_mean - outs_mean.min(axis = 0))/(outs_mean.max(axis = 0) - outs_mean.min(axis = 0))
    # epistemic_uncertainty = entropy(outs_mean)
    epistemic_uncertainty = np.mean(outs**2, axis=0) - np.mean(outs, axis=0)**2
    mean_epi_uncertainty = np.mean(epistemic_uncertainty)
    aleatoric_uncertainty = -np.mean(outs*(1-outs), axis=0)
    mean_ale_uncertainty = np.mean(aleatoric_uncertainty)
    print(mean_epi_uncertainty)
    # Open file for writing in append mode
    with open(str(mc_samples) + 'MCSamples_dropout' + str(p) + '_UQ_varianceFormulation.txt' , 'a') as file:
        file.write('Model with ' + str(n_train_samples) + ' examples:' + '\n')
        file.write("Mean epistemic uncertainty: " + str(mean_epi_uncertainty) + "\n")
        file.write("Mean aleatoric uncertainty: " + str(mean_ale_uncertainty) + "\n")
        file.write("---------------------------------------------------------------- \n")
        file.close()


# view data
fig, ax = plt.subplots(figsize=(12,7))
plt.scatter(x[:,0], x[:,1], c = y)
plt.scatter(x_val[:,0], x_val[0:,1], c = y_val)
plt.savefig('temp.jpg')
plt.close()