import os
import argparse
import time as t
import torch  # Import main library
from torch.utils.data import DataLoader  # Main class for threaded data loading
import matplotlib.pyplot as plt
import numpy as np
from utils.compute_global_acc import compute_acc
from utils.export_results import pkl_export, pkl_concat


parser = argparse.ArgumentParser(description='NN competition')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.05, dest='learning_rate', help='Optimizer learning rate')
parser.add_argument('--batch_size', default=100, type=int, help='Number of samples used to estimate the gradient (bigger = more stable training & bigger learning rate can be used)')
parser.add_argument('--concat', action='store_true', help='flag to decide wheter the concat the pickle files')
parser.add_argument('--target_class', required=True, type=int, help="Class to be learned")
parser.add_argument('--epochs', type=int, default=100, help='Number of iterations over the whole dataset.')

args = parser.parse_args()

# Optimizaiton config
target_class = args.target_class
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs =  args.epochs
concat = args.concat

val_output_path = "competition/results/val/predictions_class" + str(target_class) + ".pkl" # path to store the intermediate validation predictions
test_output_path = "competition/results/test/predictions_class" + str(target_class) + ".pkl" # path to store the intermediate test predictions
final_output_path = "competition/results/final" # path to store the final predictions

# Prepare data
train_data = np.load('competition/train.npy', allow_pickle=True)
val_data = np.load('competition/val.npy', allow_pickle=True)
test_data = np.load('competition/test_alumnes.npy', allow_pickle=True)

def select_class(data, clss, train=False):
    images = np.array(data.item()["images"])
    labels = np.array(data.item()["labels"])
    indices = np.arange(labels.shape[0])

    if train:
        np.random.shuffle(indices)

    images = images[indices]
    labels = labels[indices]
    labels = (labels == target_class).astype(int)
    return images, labels

train_images, train_labels = select_class(train_data, target_class, train=True)
val_images, val_labels = select_class(val_data, target_class, train=False)

train_size = train_labels.shape[0]
val_size = val_labels.shape[0]

# Check that the images have been loaded correctly
indices = np.arange(train_size)
positive_indices = indices[train_labels == 1]
negative_indices = indices[train_labels == 0]

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()  # Necessary for torch to detect this class as trainable
        # Here define network architecture
        self.layer1 = torch.nn.Linear(28**2, 4) # Linear layer with 4 neurons
        self.layer2 = torch.nn.Linear(4, 4) # Linear layer with 4 neurons
        self.output = torch.nn.Linear(4, 1) # Linear layer with 1 neuron (binary output)

    def forward(self, x):
        # Here define architecture behavior
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return torch.sigmoid(self.output(x)) # Binary output


# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate network
model = NeuralNet().to(device)

# Create optimizer for the network parameters
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

# Instantiate loss function
criterion = torch.nn.BCELoss()  # Binary logistic regression


# Function to iterate the training set and update network weights with batches of images.
def train(model, optimizer, criterion):

    model.train()  # training mode

    running_loss = 0
    running_corrects = 0
    total = 0

    for idx in range(0, train_size, batch_size):
        optimizer.zero_grad()  # make the gradients 0
        x = torch.from_numpy(train_images[idx:(idx + batch_size), ...]).float().to(device)
        y = torch.from_numpy(train_labels[idx:(idx + batch_size), ...]).float().to(device)
        output = model(x.view(-1, 28 ** 2))  # forward pass
        preds = (output > 0.5).float()

        loss = criterion(output.view_as(y), y)  # calculate the loss value

        loss.backward() # compute the gradients
        optimizer.step() # uptade network parameters

        # statistics
        running_loss += loss.item() * x.size(0)
        # .item() converts type from torch to python float or int
        running_corrects += torch.sum(preds.data.view(-1)==y.data.view(-1)).item()
        total += float(y.size(0))

    epoch_loss = running_loss / total  # mean epoch loss
    epoch_acc = running_corrects / total  # mean epoch accuracy

    return epoch_loss, epoch_acc


# Function to iterate the validation set and update network weights with batches of images.
def val(model, criterion):

    model.eval()  # validation mode

    running_loss = 0
    running_corrects = 0
    total = 0

    ret = []
    # We are not backpropagating trhough the validation set, so we can save time
    with torch.no_grad():
        for idx in range(0, val_size, batch_size):
            x = torch.from_numpy(val_images[idx:(idx + batch_size), ...]).float().to(device)
            y = torch.from_numpy(val_labels[idx:(idx + batch_size), ...]).float().to(device)
            output = model(x.view(-1, 28 ** 2))  # forward pass
            preds = (output > 0.5).float()
            loss = criterion(output.view_as(y), y)  # calculate the loss value

            # statistics
            running_loss += loss.item() * x.size(0)
            # .item() converts type from torch to python float or int
            running_corrects += torch.sum(preds.data.view(-1)==y.data.view(-1)).item()
            total += float(y.size(0))
            ret += (output.cpu().numpy().flatten().tolist())

    epoch_loss = running_loss / total # mean epoch loss
    epoch_acc = running_corrects / total # mean epoch accuracy
    pkl_export(ret, val_output_path)
    return epoch_loss, epoch_acc


def predict(model, output_path=test_output_path):

    test_images, _ = select_class(test_data, target_class)

    ret = []
    with torch.no_grad():
        for sample in test_images:

            x = torch.from_numpy(sample).float().to(device)
            output = model(x.view(-1, 28**2))
            _, preds = torch.max(output, 1)
            ret += (output.cpu().numpy().flatten().tolist())
    pkl_export(ret, test_output_path)


for epoch in range(epochs):

    start = t.time()

    train_loss, train_acc = train(model, optimizer, criterion)
    print('-' * 74)
    print('| End of epoch: {:3d} | Time: {:.2f}s | Train loss: {:.3f} | Train acc: {:.3f}|'
          .format(epoch + 1, t.time() - start, train_loss, train_acc))

    start = t.time()
    val_loss, val_acc = val(model, criterion)
    print('-' * 74)
    print('| End of epoch: {:3d} | Time: {:.2f}s | Val loss: {:.3f} | Val acc: {:.3f}|'
          .format(epoch + 1, t.time() - start, val_loss, val_acc))

predict(model)

if concat:
    pkl_concat("competition/results/val/", os.path.join(final_output_path, 'val_preds.pkl'))
    compute_acc(os.path.join(final_output_path, 'val_preds.pkl'))
    pkl_concat("competition/results/test/", os.path.join(final_output_path, 'predictions_class.pkl'))
