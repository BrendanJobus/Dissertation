import os
from FEMNIST_Balanced_DC import run_balanced_EMNIST_DC
from FEMNIST_Imbalanced_DC import run_imbalanced_EMNIST_DC
from CIFAR_Balanced_DC import run_balanced_CIFAR_DC
from CIFAR_Imbalanced_DC import run_imbalanced_CIFAR_DC

number_of_runs_per_dataset = 5

datasets = ["FEMNIST_balanced", "FEMNIST_imbalanced", "CIFAR_balanced", "CIFAR_imbalanced"]
dataset_method = [run_balanced_EMNIST_DC, run_imbalanced_EMNIST_DC, run_balanced_CIFAR_DC, run_imbalanced_CIFAR_DC]
conf_files = [os.path.abspath("conf/EMNIST_balance_drift_correction_conf.json"), os.path.abspath("conf/EMNIST_imbalance_drift_correction_conf.json"), os.path.abspath("conf/CIFAR_balance_drift_correction_conf.json"), os.path.abspath("conf/CIFAR_imbalance_drift_correction_conf.json")]

for dataset, dataset_method, conf in zip(datasets, dataset_method, conf_files):
    for run in range(number_of_runs_per_dataset):
        print("Currently on run {} of {} for {} dataset".format(run + 1, number_of_runs_per_dataset, dataset))
        dataset_method(conf)