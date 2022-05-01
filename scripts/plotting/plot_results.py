import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# unfortunately this is required to use relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(
                os.path.dirname(SCRIPT_DIR)))

from scripts.solver import Solver

"""Create a 2x1 grid of subplots showing training and validation data
side by side. Use to plot loss or accuracy over batch number."""
def plot_loss_and_acc(loss, acc, window=100):

    plot_loss = np.array([])
    plot_acc = np.array([])

    plot_loss_mean = np.array([])
    plot_acc_mean = np.array([])

    for i in range(len(loss)):
        plot_loss = np.append(plot_loss, loss[i])
        plot_acc   = np.append(plot_acc, acc[i])

    for i in range(1,np.shape(plot_loss)[0]):
        if i < window:
            plot_loss_mean = np.append(plot_loss_mean, np.mean(plot_loss[:i]))
            plot_acc_mean   = np.append(plot_acc_mean, np.mean(plot_acc[:i]))
        else:
            plot_loss_mean = np.append(plot_loss_mean, np.mean(plot_loss[(i-window):i]))
            plot_acc_mean   = np.append(plot_acc_mean, np.mean(plot_acc[(i-window):i]))

    xvals_loss = np.arange(np.shape(plot_loss)[0])
    xvals_loss_mean = np.arange(np.shape(plot_loss_mean)[0])

    xvals_acc = np.arange(np.shape(plot_acc)[0])
    xvals_acc_mean = np.arange(np.shape(plot_acc_mean)[0])

    fig = plt.figure()

    fig.add_subplot(2,1,1)
    plt.title("Loss")
    plt.xlabel("Batch #")
    plt.ylabel("Loss")
    plt.plot(xvals_loss, plot_loss, label='Loss', color='b',alpha=0.5, linewidth=0.5)
    plt.plot(xvals_loss_mean, plot_loss_mean, label='Mean Loss', color='r')
    plt.grid(True)
    plt.ylim([np.min(plot_loss), np.max(plot_loss)])
    plt.yticks(np.linspace(0, np.max(plot_loss), num=5))
    plt.xlim([0, xvals_loss[-1]])
    plt.legend(loc='upper right', ncol=2, fontsize=8)

    fig.add_subplot(2,1,2)
    plt.title("Accuracy")
    plt.xlabel("Batch #")
    plt.ylabel("Loss")
    plt.plot(xvals_acc, plot_acc, label='Accuracy', color='b',alpha=0.5, linewidth=0.5)
    plt.plot(xvals_acc_mean, plot_acc_mean, label='Mean Accuracy', color='r')
    plt.grid(True)
    plt.ylim([np.min(plot_acc), np.max(plot_acc)])
    plt.yticks(np.linspace(0, np.max(plot_acc), num=5))
    plt.xlim([0, xvals_acc[-1]])
    plt.legend(loc='lower right', ncol=2, fontsize=8)

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    from torch import load
    s_dict = load(os.path.join(SCRIPT_DIR,'..','training','checkpoints','example_trained_model.pt'))
    solver = s_dict['solver']

    plot_loss_and_acc(solver.train_loss_history, solver.train_acc_history)
    plot_loss_and_acc(solver.val_loss_history, solver.val_acc_history)