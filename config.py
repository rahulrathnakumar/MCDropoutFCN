import argparse


# Optional arguments : Modifiables for shell script - training set size only for now
parser = argparse.ArgumentParser()
# parser.add_argument("--root_dir", 
# help = "Specify training data path Eg: '/home/rrathnak/Documents/Work/Task-2/Datasets/asu_cropped'")
# parser.add_argument("--dataset")
parser.add_argument("--num_training", help = 'Number of training samples')
args = parser.parse_args()


num_epochs = 1500
root_dir = '/home/rrathnak/Documents/Work/Task-2/Datasets/RoadCracks'
dataset = 'RoadCracks'
is_rgbd = False
val_dataset = 'ASU'
num_classes = 2
batch_size = 8
lr = 0.01
momentum = 0.9
optim_w_decay = 1e-5
step_size = 1500
gamma = 0.1
load_ckp = False
print_gpu_usage = False
dropout_prob = 0.1
mc_samples = 5
num_training = args.num_training
repeats = 4
optimizer_name = 'SGD'
directory_name = str(dataset) + '_' + str(dropout_prob) + 'dropout_'+ str(num_training) + 'Train' + '_StepLR'
save_dir_name = 'Training_' + directory_name + '_' + 'Val_' + str(val_dataset)
print(args.num_training)