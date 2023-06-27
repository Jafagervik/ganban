import numpy as np 
import matplotlib.pyplot as plt

def plot_acc_loss(results):
    """
    Training data statistics
    """
    x = np.array([i for i in range(len(results["0"]["train_acc"]))])

    plt.plot(x, results["0"]["train_acc"], color='b', label="Train acc")
    plt.plot(x, results["0"]["train_loss"], color='r', label="Train loss")
    plt.plot(x, results["0"]["test_acc"], color='g', label="Test acc")
    plt.plot(x, results["0"]["test_loss"], color='m', label="Test loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss / Acc")
    plt.title("Statistics")

    plt.legend()

    plt.xlim([1, len(x)])
    plt.ylim([0.0, 100.0])

    plt.show()

