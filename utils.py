import torch
import shutil
import numpy as np
from visdom import Visdom
import matplotlib.pyplot as plt
import imshowpair

SMOOTH = 1e-6

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

    def image(self, var_name, title_name, img):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.image(img = img, env = self.env, opts = dict(
                title = title_name,
            ))
        else:
            self.viz.image(img = img, env=self.env, win=self.plots[var_name])

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    
    torch.save(state, checkpoint_dir + 'checkpoint.pt')
    if is_best:
        best_fpath = best_model_dir +'/best_model.pt'
        shutil.copyfile(checkpoint_dir + 'checkpoint.pt', best_fpath)


def load_ckp(checkpoint_fpath, net, optimizer = None):
    checkpoint = torch.load(checkpoint_fpath)
    net.load_state_dict(checkpoint['net_state_dict'])
    if optimizer is not None: 
        optimizer.load_state_dict(checkpoint['optimizer'])
        return net, optimizer, checkpoint['epoch']
    return net, checkpoint['epoch']
    

def save_predictions(imgList, path):
    # numImages = len(imgList)
    # fig = plt.figure(figsize=(8,2))
    # for i in range(0,numImages):
    #     plt.subplot(1,numImages, i+1)
    #     plt.imsave(imgList[i], aspect='auto')
    # plt.tight_layout()
    # plt.savefig(path)
    # plt.close()
    final = np.concatenate(imgList, axis = 1)
    plt.imsave(path, final)

def pixel_accuracy(output, target):
    accuracy = []
    N, C, h, w = output.shape
    output = output.data.cpu().numpy()
    pred = output.transpose(0, 2, 3, 1).reshape(-1, C).argmax(axis=1).reshape(N, h, w)
    target = target.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, C).argmax(axis=1).reshape(N, h, w)
    for p,t in zip(pred, target):
        correct = (p == t).sum()
        total   = (t == t).sum()
        accuracy.append(correct/total)
    
    return accuracy

def f1(iou):
    dice = (2*iou)/(1+ iou)
    return dice


def iou(output, target, per_image = False):
    N, C, h, w = output.shape
    output = output.data.cpu().numpy()
    pred = output.transpose(0, 2, 3, 1).reshape(-1, C).argmax(axis=1).reshape(N, h, w)
    predIU = np.zeros((pred.shape[0], C, h, w))
    for i in range(pred.shape[0]):
        for c in range(C):
            predIU[i][c][pred[i] == c] = 1
    
    predIU = predIU.astype(np.uint8)
    target = target.cpu().numpy()
    target = target.astype(np.uint8)
    if per_image:
        # batchIOU = classIU(predIU, target)[:,1]
        batchIOU = np.nanmean(classIU(predIU, target), axis = 1)
        return batchIOU
    else:
        batchIOU = (np.nanmean(classIU(predIU, target)))
        return batchIOU


def classIU(pred, target):
    N,c,_,_ = pred.shape
    iou = np.zeros((N,c))
    for i in range(N):
        for j,(p,t) in enumerate(zip(pred[i],target[i])):
            intersection = (p & t).sum()
            union = (p | t).sum()
            if intersection == 0 and union == 0:
                iou[i,j] = 'nan'
            else:
                iou[i,j]= ((intersection + SMOOTH)/(union + SMOOTH))
    return iou

def normalize(img):
    norm_img = (img - np.min(img))/np.ptp(img)
    norm_img[np.isnan(norm_img)] = 0
    return norm_img



# Epistemic Uncertainty
def entropy(y):
    N,C,H,W = y.shape
    entropy_c = []
    for c in range(C):
        entropy_c.append(y[:,c,:,:]*np.log(y[:,c,:,:] + 1e-10))
    uncertainty = -np.sum(np.asarray(entropy_c), axis = 0)
    # uncertainty = np.asarray(entropy_c)[1]
    return uncertainty

def stochastic_log_softmax(mean, var):
    '''
    Returns Lx in the paper
    '''
    # Sample 't' times from a normal distribution to get x_hat
    x_hats = list()
    for t in range(10):
        eps = torch.randn(var.size()).cuda()
        x_hat = mean + torch.mul(var, eps)
        x_hats.append(x_hat)
    x_hats = torch.stack(x_hats)
    return torch.mean(torch.log_softmax(x_hats , dim = 2), dim = 0)
