import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchviz import make_dot

def visualize_model (model: torch.nn.Module,
                     yhat):
    """
    creates a graph of the model with an input of yhat
    makes use of torchviz:
    https://github.com/szagoruyko/pytorchviz

    Args:
        model (torch.nn.Module): model
        yhat: input to use
    """
    return make_dot(yhat, params=dict(list(model.named_parameters()))).render("model-viz", format="png")

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))
    plt.title("Loss and Accuracy of Training and Testing")

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="training loss")
    plt.plot(epochs, test_loss, label="testing loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="training accuracy")
    plt.plot(epochs, test_accuracy, label="testing accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))
    # Plot training data
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    # Plot test data
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    # Plot the predictions, if any
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})