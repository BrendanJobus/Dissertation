import unittest
import numpy as np
import os
import pickle

class TestVariableSetting(unittest.TestCase):
    def test_initialize(self):
        pass

    def test_mask(self):
        pass

    def test_global_model(self):
        pass

    def test_history_update(self):
        pass

class TestDriftCorrection(unittest.TestCase):
    def test_drift_function(self):
        pass

    def test_loss_cp(self):
        pass

    def test_objective_function(self):
        pass

    def test_model_cloning(self):
        pass

    def test_data_save(self):
        save_dir_path = "result_FEMNIST_balanced_test"
        model_type="base"
        test_dicts = [{0.012345}, {0.012345, 0.012334567}, {0.012345, 0.012334567}, {0.012345, 0.012334567}, {0.012345, 0.012334567}]
        with open(os.path.join(save_dir_path, 'pooled_train_result_{}.pkl'.format(model_type)), 'wb') as f:
            pickle.dump(test_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)

class TestCompareSetup(unittest.TestCase):
    def test_performance_recording(self):
        number_of_tests = 5
        test_performance = np.zeros(shape=(number_of_tests), dtype=dict)
        test_dicts = [{0.012345}, {0.012345, 0.012334567}, {0.012345, 0.012334567}, {0.012345, 0.012334567}, {0.012345, 0.012334567}]
        correct_performance = np.array([{0.012345}, {0.012345, 0.012334567}, {0.012345, 0.012334567}, {0.012345, 0.012334567}, {0.012345, 0.012334567}])

        for i in range(number_of_tests):
            test_performance[i] = test_dicts[i]

        self.assertEqual(test_performance.all(), correct_performance.all())

    def test_model_used(self):
        pass

    def test_performance_analytics(self):
        pass

if __name__ == '__main__':
    unittest.main()