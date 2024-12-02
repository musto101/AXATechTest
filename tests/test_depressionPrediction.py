# a testing framework for the depressionPrediction.py script

from src.depressionPrediction import DepressionTask
import pandas as pd
import matplotlib.pyplot as plt
import os

# load the data

file_path = 'data/depression/depression_data.csv'
depression_task = DepressionTask(file_path)
data = depression_task.load_data()
assert isinstance(data, pd.DataFrame), "The data should be a pandas DataFrame"

# test the preprocess_data method
X, Y = depression_task.preprocess_data(data)
assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame"
assert isinstance(Y, pd.Series), "Y should be a pandas Series"
assert X.shape[0] == Y.shape[0], "X and Y should have the same number of rows"

# test the split_data method
X_train, X_test, Y_train, Y_test = depression_task.split_data(X, Y)
assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame"
assert isinstance(X_test, pd.DataFrame), "X_test should be a pandas DataFrame"
assert isinstance(Y_train, pd.Series), "Y_train should be a pandas Series"
assert isinstance(Y_test, pd.Series), "Y_test should be a pandas Series"

# test the standardize_data method
X_train, X_test = depression_task.standardize_data(X_train, X_test)
assert X_train.mean().sum().round() == 0, "X_train should be standardized"
assert X_test.mean().sum().round() == 0, "X_test should be standardized"

# test the balance_data method
X_train_balanced, Y_train_balanced = depression_task.balance_data(X_train, Y_train)
Y_train_value_counts = Y_train_balanced.value_counts(normalize=True)
assert Y_train_value_counts[0] == Y_train_value_counts[1] == 0.5, "The balanced data should have equal class distribution"

# test the train_rf_model method
roc_auc_rf = depression_task.train_rf_model(X_train_balanced, Y_train_balanced, X_test, Y_test)
assert isinstance(roc_auc_rf, float), "roc_auc_rf should be a float"
assert 0 <= roc_auc_rf <= 1, "roc_auc_rf should be between 0 and 1"

# test the train_lr_model method
roc_auc_lr = depression_task.train_lr_model(X_train_balanced, Y_train_balanced, X_test, Y_test)
assert isinstance(roc_auc_lr, float), "roc_auc_lr should be a float"
assert 0 <= roc_auc_rf <= 1, "roc_auc_rf should be between 0 and 1"


print("All tests passed!")