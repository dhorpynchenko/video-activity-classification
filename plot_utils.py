from matplotlib import pyplot as plt, ticker
import os


def create_plot(accuracy, loss, save_path=None, show=True, accuracy_comment=None):

    plt.subplot(2, 1, 1)
    plt.plot(loss, color='r', label='Train Loss')
    plt.grid()

    plt.subplot(2, 1, 2)
    if accuracy_comment is not None:
        plt.title(accuracy_comment)
    plt.plot(accuracy, color='b', label='Validation Accuracy')
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
