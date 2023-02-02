import os
import time
import numpy as np
import needle.nn as nn
import needle as ndl
import sys
sys.path.append('../python')

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    # BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim))
        ),
        nn.ReLU()
    )
    # END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    # BEGIN YOUR SOLUTION
    ls = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        ls.append(
            ResidualBlock(hidden_dim, hidden_dim//2,
                          norm, drop_prob)
        )
    ls.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*ls)
    # END YOUR SOLUTION


def epoch(dataloader: ndl.data.DataLoader,
          model: ndl.nn.Module,
          opt: ndl.optim.Optimizer = None):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION

    def loss_err(h, y):
        """ Helper function to compute both loss and error"""
        lossModule = nn.SoftmaxLoss()
        return (
            lossModule.forward(h, y),
            np.sum(h.numpy().argmax(axis=1) != y.numpy(), dtype=np.float32)
        )

    if opt is None:
        model.eval()
        loss = 0
        err = 0
        i = 0
        num_sample = 0
        for _, data in enumerate(dataloader):
            imgs = data[0]
            labels = data[1]
            i += 1
            forwardRes = model.forward(imgs)
            iLoss, iError = loss_err(forwardRes, labels)
            loss += iLoss.numpy()
            err += iError
            num_sample += labels.shape[0]
        return (err/num_sample, loss/i)
    else:
        model.train()
        loss = 0
        err = 0
        i = 0
        num_sample = 0
        for _, data in enumerate(dataloader):
            imgs = data[0]
            labels = data[1]
            i += 1
            forwardRes = model.forward(imgs)
            iLoss, iError = loss_err(forwardRes, labels)
            loss += iLoss.numpy()
            err += iError
            num_sample += labels.shape[0]
            iLoss.backward()
            opt.step()
        return (err/num_sample, loss/i)

    # END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        data_dir+"/train-images-idx3-ubyte.gz",
        data_dir+"/train-labels-idx1-ubyte.gz"
    )
    test_dataset = ndl.data.MNISTDataset(
        data_dir+"/t10k-images-idx3-ubyte.gz",
        data_dir+"/t10k-labels-idx1-ubyte.gz"
    )
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    train_acc, train_loss = None, None
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_acc, train_loss = epoch(train_dataloader,model=model,opt=opt)
    test_acc, test_loss = epoch(test_dataloader,model=model)

    return (train_acc, train_loss, test_acc, test_loss)
    # END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")

