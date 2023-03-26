import matplotlib.pyplot as plt


def plot_loss(error_rate1, error_rate2, parameters, parameter_name):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Plot of the error rate by parameter for pretrained network
    axes[0].plot(parameters, error_rate1)
    axes[0].set_title("Impact of number of layers on pretrained network")
    axes[0].set_xlabel(parameter_name)
    axes[0].set_ylabel("Error rate")

    # Plot of the error rate by parameter for neural network
    axes[1].plot(parameters, error_rate2)
    axes[1].set_title("Impact of number of layers on neural network")
    axes[1].set_xlabel(parameter_name)
    axes[1].set_ylabel("Error rate")

    plt.suptitle(f"Comparing error rate of both models by {parameter_name}")
    plt.show()
