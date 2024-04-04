import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def load_data():
    dataset_directory = ["result_FEMNIST_balanced_DC/", "result_FEMNIST_imbalanced_DC/", "result_CIFAR_balanced_DC/", "result_CIFAR_imbalanced_DC/"]
    base_file_name = "/col_performance_base.pkl"
    DC_file_name =   "/col_performance_drift_correct.pkl"

    paths = []

    for directory in dataset_directory:
        for d in os.walk(directory):
            if d[1]:
                paths.append(d[1])

    # save data in 2d array [[base_data, dc_data], [base_data, dc_data], [base_data, dc_data], [base_data, dc_data]]
    data = [[[], []], [[], []], [[], []], [[], []]]

    for dir, sub, d in zip(dataset_directory, paths, data):
        for s in sub:
            base_data_path = os.path.abspath(dir + s + base_file_name)
            with open(base_data_path, 'rb') as f:
                loaded_data = pickle.load(f)
                loaded_data = [v for _, v in loaded_data.items()]
                d[0].append(loaded_data)

            dc_data_path = os.path.abspath(dir + s + DC_file_name)
            with open(dc_data_path, 'rb') as f:
                loaded_data = pickle.load(f)
                loaded_data = [v for _, v in loaded_data.items()]
                d[1].append(loaded_data)

    return data[0], data[1], data[2], data[3], data

dataset_names = ["EMNIST", "CIFAR"]

def plot_alone(data, title):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True, sharey = True)
    legend = ["model 0", "model 1", "model 2", "model 3", "model 4", "model 5", "model 6", "model 7", "model 8", "model 9"]


def plot_average_and_variance(datasets):
    titles = ["IID EMNIST", "Non-IID EMNIST", "IID CIFAR", "Non-IID CIFAR"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True, sharey=True)
    axs = [ax1, ax2, ax3, ax4]

    for dataset, title, axis in zip(datasets, titles, axs):
        base_data = dataset[0]; dc_data = dataset[1]
        base_model_means = []
        for model in base_data:
            base_model_means.append(np.mean(model, axis=0))
        base_data_mean = np.mean(base_model_means, axis=0)
        base_data_err = np.var(base_model_means, axis=0)
        axis.errorbar(range(len(base_data_mean)), base_data_mean, base_data_err)
        

        dc_model_means = []
        for model in dc_data:
            dc_model_means.append(np.mean(model, axis=0))
        dc_data_mean = np.mean(dc_model_means, axis=0)
        dc_data_err = np.var(dc_model_means, axis=0)
        axis.errorbar(range(len(dc_data_mean)), dc_data_mean, dc_data_err)

        axis.set_title(title)
        axis.set(xlabel = 'Epoch', ylabel = 'Accuracy')

    fig.legend(["base", "drift corrected"])
    fig.suptitle("Mean and variance of the two implementations")
    plt.show()


emnist_balanced, emnist_imbalanced, cifar_balanced, cifar_imbalanced, datasets = load_data()

#plot_alone(emnist_balanced, "EMNIST data set IID")
plot_average_and_variance(datasets)