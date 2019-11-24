import torch  # Import main library
from torch.utils.data import DataLoader  # Main class for threaded data loading
import torch.nn.functional as func
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from IPython import display
import seaborn as sb
import time

# Load datasets
train_data = np.load('../base_dades_xarxes_neurals/train.npy', allow_pickle=True)
val_data = np.load('../base_dades_xarxes_neurals/val.npy', allow_pickle=True)


def select_class(data, clss, target_class=1):
    images = np.array(data.item()["images"])
    labels = np.array(data.item()["labels"])
    labels = (labels == target_class).astype(int)  # Binarització etiqueta 0 si != target, si 1 == target
    return images, labels


def getAllData(data, train=False):
    images = np.array(data.item()["images"])
    labels = np.array(data.item()["labels"])
    if train:
        indices = list(range(images.shape[0]))
        random.shuffle(indices)  # al passar les mostres de test a Codalab, NO FER SHUFFLE
        images = images[indices]
        labels = labels[indices]
    return images, labels


# Creació xarxa neuronal
def init_weights(net):
    if type(net) == torch.nn.Module:
        torch.nn.init.xavier_uniform_(net.weight)  # Xavier initialization
        net.bias.data.fill_(0.01)  # tots els bias a 0.01


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()  # Necessary for torch to detect this class as trainable
        # Here define network architecture
        self.layer1 = torch.nn.Linear(28 ** 2, 32)  # Linear layer with 32 neurons
        self.layer2 = torch.nn.Linear(32, 64)  # Linear layer with 64 neurons

        # Apartat B, C
        self.output = torch.nn.Linear(64, 1)  # Linear layer with 1 neuron (binary output)

        # Apartat A
        # self.output = torch.nn.Linear(64, 10) # Linear layer with 10 neurons (10-class output)

    def forward(self, x):
        # Here define architecture behavior
        x = torch.sigmoid(self.layer1(x))  # x = torch.nn.functional.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))  # si dona error 'Cannot find reference 'sigmoid' in '__init__.py',
        # proveu x = func.sigmoid(self.layer1(x)), havent fet: import torch.nn.functional as func

        # Apartat B,C
        return torch.sigmoid(self.output(x))  # Binary output

        # Apartat A
        # return torch.nn.functional.log_softmax(self.output(x), dim=1)  # 10 classes neural network


# Optimizaiton config
target_class = [i for i in range(1, 10)]  # Train a classifier for this class
batch_size = 750  # Number of samples used to estimate the gradient (bigger = stable training & bigger learning rate)
learning_rate = 0.05  # Optimizer learning rate
epochs = 250  # Number of iterations over the whole dataset.

train_img = []
train_lb = []
val_img = []
val_lb = []
train_size = []
val_size = []
models = []
optimizers = []

for target in target_class:

    # Entrenament i validació apartat C i B
    train_images, train_labels = select_class(train_data, target)  # Binary case: here class 3
    train_img.append(train_images)
    train_lb.append(train_labels)

    val_images, val_labels = select_class(val_data, target)  # Binary case: here class 3
    val_img.append(val_images)
    val_lb.append(val_labels)

    # Entrenament i validació apartat A (descommentar)
    # train_images, train_labels = getAllData(train_data, train=True) # 10-class case (Apartat A)
    # val_images, val_labels = getAllData(val_data, train=False) # 10-class case (Apartat A)

    train_size.append(train_labels.shape[0])
    val_size.append(val_labels.shape[0])

    # print(train_size, "training images.")

    # Verificació data load
    indices = np.arange(train_size[target-1])
    positive_indices = indices[train_labels == 1]
    negative_indices = indices[train_labels == 0]

    # Uncomment to show image examples from ds
    """
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Positive")
    plt.imshow(train_images[positive_indices[0], :, :], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Negative")
    plt.imshow(train_images[negative_indices[0], :, :], cmap="gray")
    """

    # Instantiate network
    models.append(NeuralNet())

    # Creem l'optimitzador, declarem la funció a optimitzar, i la resta de funcions auxiliars per a optimitzar el model
    # Try different initialization techniques, en aquest cas la de Xavier
    models[target-1].apply(init_weights)

    # Create optimizer for the network parameters
    optimizers.append(torch.optim.SGD(models[target-1].parameters(), learning_rate))

    # Instantiate loss function
    # criterion = torch.nn.BCELoss()  # Binary logistic regression
    criterion = torch.nn.MSELoss()


# Function to iterate the training set and update network weights with batches of images.
def train(model, optimizer, criterion, target, batch_size=batch_size):
    model.train()  # training model

    running_loss = 0
    running_corrects = 0
    total = 0

    for idx in range(0, train_size[target-1], batch_size):
        optimizers[target-1].zero_grad()  # make the gradients 0
        x = torch.from_numpy(train_img[target-1][idx:(idx + batch_size), ...]).float()
        y = torch.from_numpy(train_lb[target-1][idx:(idx + batch_size), ...]).float()
        output = model(x.view(-1, 28 ** 2))  # forward pass

        # Apartat B,C
        preds = (output > 0.5).float()
        loss = criterion(output.view_as(y), y)  # calculate the loss value

        # Apartat A
        # 10 classes neural network (Apartat A)
        # preds = torch.argmax(output, 1)
        # loss = torch.nn.functional.cross_entropy(output, y.long())

        loss.backward()  # compute the gradients
        optimizers[target-1].step()  # uptade network parameters

        # statistics
        running_loss += loss.item() * x.size(0)

        # Apartat B,C
        running_corrects += torch.sum(
            preds.data.view(-1) == y.data.view(-1)).item()  # .item() converts type from torch to python float or int

        # La raó és que preds té mida (batch, 1) i labels té mida (batch).
        # Pel què al fer ==, torch repeteix preds i labels per a que tinguin mida (batch, batch) i coincideixin
        # el .data, l'únic que fa es dir a torch que no cal fer backpropagation amb això.

        # Apartat A
        # running_corrects += torch.sum(preds == y.data.view(-1).long()).item()

        total += float(y.size(0))

    epoch_loss = running_loss / total  # mean epoch loss
    epoch_acc = running_corrects / total  # mean epoch accuracy

    return epoch_loss, epoch_acc


# Function to iterate the validation set and update network weights with batches of images.
def val(model, criterion, target):
    model.eval()  # validation mode

    running_loss = 0
    running_corrects = 0
    total = 0

    predicted = []

    with torch.no_grad():  # We are not backpropagating trhough the validation set, so we can save speed
        for idx in range(0, val_size[target-1], batch_size):
            x = torch.from_numpy(val_img[target-1][idx:(idx + batch_size), ...]).float()
            y = torch.from_numpy(val_lb[target-1][idx:(idx + batch_size), ...]).float()
            output = model(x.view(-1, 28 ** 2))  # forward pass

            # Apartat B,C
            # Binary case (Apartat B, C)
            preds = (output > 0.5).float()
            loss = criterion(output.view_as(y), y)  # calculate the loss value

            # Apartat A
            # 10-classes nbeural network (Apartat A)
            # preds = torch.argmax(output, 1)
            # loss = torch.nn.functional.cross_entropy(output, y.long())

            predicted += preds.tolist()

            # statistics
            running_loss += loss.item() * x.size(0)

            # Apartat B,C
            running_corrects += torch.sum(preds.data.view(-1) == y.data.view(
                -1)).item()  # .item() converts type from torch to python float or int

            # La raó és que preds té mida (batch, 1) i labels té mida (batch).
            # Pel què al fer ==, torch repeteix preds i labels per a que tinguin mida (batch, batch) i coincideixin
            # el .data, l'únic que fa es dir a torch que no cal fer backpropagation amb això.

            # Apartat A
            # running_corrects += torch.sum(preds == y.data.view(-1).long()).item()

            total += float(y.size(0))

    targets = np.array(val_lb[target-1])
    confusionMatrix = confusion_matrix(targets, np.array(predicted))

    epoch_loss = running_loss / total  # mean epoch loss
    epoch_acc = running_corrects / total  # mean epoch accuracy

    return epoch_loss, epoch_acc, confusionMatrix


# Funció per evaluar el comportament del model en funció del learning rate
def entrenamentMain(target):
    # Loop d'entrenament principal
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    # Remove this line out of jupyter notebooks
    for epoch in range(epochs):
        t_loss, t_acc = train(models[target-1], optimizers[target-1], criterion, target)

        v_loss, v_acc, confusionMatrix = val(models[target-1], criterion, target)
        # print(confusionMatrix)

        train_loss.append(t_loss)
        train_accuracy.append(t_acc)
        val_loss.append(v_loss)
        val_accuracy.append(v_acc)
        plt.title("Results for target class %s" % target)
        plt.subplot(1, 2, 1)
        plt.title("loss")
        plt.plot(train_loss, 'b-')
        plt.plot(val_loss, 'r-')
        plt.legend(["train", "val"])
        plt.subplot(1, 2, 2)
        plt.title("accuracy")
        plt.plot(train_accuracy, 'b-')
        plt.plot(val_accuracy, 'r-')
        plt.legend(["train", "val"])
        display.clear_output(wait=True)
        # display.display(plt.gcf())
        if epoch == range(epochs)[-1]:
            plt.show()

            plt.figure()
            plt.title("Confusion Matrix Classifier %s" % target)
            ax = sb.heatmap(confusionMatrix, cmap="Blues", annot=True, fmt="d")
            plt.show()

    display.clear_output(wait=True)


def learningRateTest(target):
    learning_rates = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    exe_time = []

    for lr in learning_rates:
        t_ini = time.time()
        optimizers[target-1] = torch.optim.SGD(models[target-1].parameters(), lr)

        # Remove this line out of jupyter notebooks
        for epoch in range(epochs):
            t_loss, t_acc = train(models[target-1], optimizers[target-1], criterion, target)
            v_loss, v_acc, confusionMatrix = val(models[target-1], criterion, target)
            # print(confusionMatrix)

            if epoch == range(epochs)[-1]:
                train_loss.append(t_loss)
                train_accuracy.append(t_acc)
                val_loss.append(v_loss)
                val_accuracy.append(v_acc)

                plt.figure()
                plt.title("Heatmap with learning rate %s" % lr)
                ax = sb.heatmap(confusionMatrix, cmap="Blues", annot=True, fmt="d")
                plt.show()

        exe_time.append(time.time() - t_ini)

    plt.subplot(1, 2, 1)
    plt.title("loss")
    plt.plot(learning_rates, train_loss, 'b-')
    plt.xscale('log')
    plt.plot(learning_rates, val_loss, 'r-')
    plt.xscale('log')
    plt.legend(["train", "val"])
    plt.subplot(1, 2, 2)
    plt.title("accuracy")
    plt.plot(learning_rates, train_accuracy, 'b-')
    plt.xscale('log')
    plt.plot(learning_rates, val_accuracy, 'r-')
    plt.xscale('log')
    plt.legend(["train", "val"])
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.show()
    plt.plot(learning_rates, exe_time, 'g-')
    plt.xscale('log')
    plt.show()
    # print("Temps d'execució total: ",time.time() - t_ini)


def epochTest(target):
    epochs_test = [5, 10, 25, 50, 75, 100]
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    exe_time = []

    for epoch in epochs_test:
        print("Training model with %s epochs..." % epoch)
        t_ini = time.time()
        optimizers[target-1] = torch.optim.SGD(models[target-1].parameters(), learning_rate)

        # Remove this line out of jupyter notebooks
        for it in range(epoch):
            t_loss, t_acc = train(models[target-1], optimizers[target-1], criterion, target)
            v_loss, v_acc, confusionMatrix = val(models[target-1], criterion, target)
            # print(confusionMatrix)

            if it == range(epoch)[-1]:
                train_loss.append(t_loss)
                train_accuracy.append(t_acc)
                val_loss.append(v_loss)
                val_accuracy.append(v_acc)

                plt.figure()
                plt.title("Heatmap with %s epochs" % epoch)
                ax = sb.heatmap(confusionMatrix, cmap="Blues", annot=True, fmt="d")
                plt.show()

        exe_time.append(time.time() - t_ini)

    plt.subplot(1, 2, 1)
    plt.title("loss")
    plt.plot(epochs_test, train_loss, 'b-')
    plt.xscale('log')
    plt.plot(epochs_test, val_loss, 'r-')
    plt.xscale('log')
    plt.legend(["train", "val"])
    plt.subplot(1, 2, 2)
    plt.title("accuracy")
    plt.plot(epochs_test, train_accuracy, 'b-')
    plt.xscale('log')
    plt.plot(epochs_test, val_accuracy, 'r-')
    plt.xscale('log')
    plt.legend(["train", "val"])
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.show()
    plt.plot(epochs_test, exe_time, 'g-')
    plt.xscale('log')
    plt.show()
    # print("Temps d'execució total: ",time.time() - t_ini)


def batchTest(target):
    batches = [10, 20, 50, 75, 100, 200, 500, 1000]
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    exe_time = []

    for batch in batches:
        t_ini = time.time()
        optimizers[target-1] = torch.optim.SGD(models[target-1].parameters(), learning_rate)

        # Remove this line out of jupyter notebooks
        for epoch in range(epochs):
            t_loss, t_acc = train(models[target-1], optimizers[target-1], criterion, batch, target)
            v_loss, v_acc, confusionMatrix = val(models[target-1], criterion, target)
            # print(confusionMatrix)

            if epoch == range(epochs)[-1]:
                train_loss.append(t_loss)
                train_accuracy.append(t_acc)
                val_loss.append(v_loss)
                val_accuracy.append(v_acc)

                plt.figure()
                plt.title("Heatmap with batch size %s" % batch)
                ax = sb.heatmap(confusionMatrix, cmap="Blues", annot=True, fmt="d")
                plt.show()

        exe_time.append(time.time() - t_ini)

    plt.subplot(1, 2, 1)
    plt.title("loss")
    plt.plot(batches, train_loss, 'b-')
    plt.xscale('log')
    plt.plot(batches, val_loss, 'r-')
    plt.xscale('log')
    plt.legend(["train", "val"])
    plt.subplot(1, 2, 2)
    plt.title("accuracy")
    plt.plot(batches, train_accuracy, 'b-')
    plt.xscale('log')
    plt.plot(batches, val_accuracy, 'r-')
    plt.xscale('log')
    plt.legend(["train", "val"])
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.show()
    plt.plot(batches, exe_time, 'g-')
    plt.xscale('log')
    plt.show()
    # print("Temps d'execució total: ",time.time() - t_ini)


for target in target_class:
    print("Calculating results for classifier %s" % target)
    entrenamentMain(target)
