import os
import numpy as np
import tensorflow as tf
from FEMNIST_Balanced import run_balanced
from FEMNIST_Balanced_DC import run_balanced_DC
from tensorflow.keras.models import Model
from Neural_Networks_DC import DriftCorrectionModel

number_of_tests = 5
base_model = Model
dc_model = DriftCorrectionModel

regular_performance = np.zeros(shape=(number_of_tests), dtype=dict)
drift_correction_performance = np.zeros(shape=(number_of_tests), dtype=dict)

fedMD_conf_file = os.path.abspath("conf/EMNIST_balance_drift_correction_conf.json")
drif_correction_conf_file = os.path.abspath("conf/EMNIST_balance_conf.json")

for i in range(number_of_tests):
    print("Running Comparison {}".format(i + 1))
    print("\nRunning standard FedMD\n")
    regular_performance[i] = run_balanced_DC(fedMD_conf_file, base_model)
    tf.keras.backend.clear_session()
    print("\nRunning drift corrected FedMD")
    tf.keras.backend.clear_session()
    print(regular_performance)
    print(drift_correction_performance)

## Need to set up the mean and variance on these, need to do it by model not by run