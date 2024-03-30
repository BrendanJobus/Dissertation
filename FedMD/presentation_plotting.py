import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset_names = ["FEMNIST", "CIFAR"]

def plot_alone(balanced_file_path, imbalanced_file_path, title):
    data = None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True, sharey=True)
    legend = ["model 0", "model 1", "model 2", "model 3", "model 4", "model 5", "model 6", "model 7", "model 8", "model 9"]

    with open(balanced_file_path.format(dataset_names[0]), 'rb') as f:
        data = pickle.load(f)
        data = [v for _, v in data.items()]
        for model in data:
            ax1.plot(range(len(model)), model)
            ax1.set_title("IID FEMNIST")
            ax1.set(xlabel='Epoch', ylabel='Accuracy')

    with open(imbalanced_file_path.format(dataset_names[0]), 'rb') as f:
        data = pickle.load(f)
        data = [v for _, v in data.items()]
        for model in data:
            ax2.plot(range(len(model)), model)
            ax2.set_title("Non IID FEMNIST")
            ax2.set(xlabel='Epoch', ylabel='Accuracy')

    with open(balanced_file_path.format(dataset_names[1]), 'rb') as f:
        data = pickle.load(f)
        data = [v for _, v in data.items()]
        for model in data:
            ax3.plot(range(len(model)), model)
            ax3.set_title("IID CIFAR")
            ax3.set(xlabel='Epoch', ylabel='Accuracy')

    with open(imbalanced_file_path.format(dataset_names[1]), 'rb') as f:
        data = pickle.load(f)
        data = [v for _, v in data.items()]
        for model in data:
            ax4.plot(range(len(model)), model)
            ax4.set_title("Non IID CIFAR")
            ax4.set(xlabel='Epoch', ylabel='Accuracy')

    fig.legend(legend, loc='lower right')
    fig.suptitle(title)
    plt.show()

def plot_average_and_variance(base_balanced, base_imbalanced, dc_balanced, dc_imbalanced):
    base_data_to_plot = [base_balanced.format(dataset_names[0]), base_imbalanced.format(dataset_names[0]), base_balanced.format(dataset_names[1]), base_imbalanced.format(dataset_names[1])]
    dc_data_to_plot = [dc_balanced.format(dataset_names[0]), dc_imbalanced.format(dataset_names[0]), dc_balanced.format(dataset_names[1]), dc_imbalanced.format(dataset_names[1])]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True)
    axs = [ax1, ax2, ax3, ax4]
    titles = ["IID FEMNIST", "Non IID FEMNIST", "IID CIFAR", "Non IID CIFAR"]
    for base_data, dc_data, ax, title in zip(base_data_to_plot, dc_data_to_plot, axs, titles):
        f = open(base_data, 'rb')
        data = pickle.load(f)
        data = [v for _, v in data.items()]
        mean = np.mean(data, axis=0)
        error = np.var(data, axis=0)
        ax.errorbar(range(len(data[0])), mean, error)
        f.close()

        f = open(dc_data, 'rb')
        data = pickle.load(f)
        data = [v for _, v in data.items()]
        mean = np.mean(data, axis=0)
        error = np.var(data, axis=0)
        ax.errorbar(range(len(data[0])), mean, error)        
        f.close()

        ax.set_title(title)
        ax.set(xlabel='Epoch', ylabel='Accuracy')

    fig.legend(["base", "drift corrected"])
    fig.suptitle("Mean and variance of the two implementations")
    plt.show()    

balanced_base_path = "presentation/result_{}_balanced_DC/col_performance_base.pkl"
imbalanced_base_path = "presentation/result_{}_imbalanced_DC/col_performance_base.pkl"
plot_alone(balanced_base_path, imbalanced_base_path, "Base Models")

balanced_dc_path = "presentation/result_{}_balanced_DC/col_performance_drift_correct.pkl"
imbalanced_dc_path = "presentation/result_{}_imbalanced_DC/col_performance_drift_correct.pkl"
plot_alone(balanced_dc_path, imbalanced_dc_path, "Drift Corrected Models")

plot_average_and_variance(balanced_base_path, imbalanced_base_path, balanced_dc_path, imbalanced_dc_path)