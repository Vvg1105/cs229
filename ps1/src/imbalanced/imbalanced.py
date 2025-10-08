import numpy as np
import util
from random import random
import os
import sys

# Ensure imports work when running as a script from src/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from imbalanced import util

### NOTE : You need to complete logreg implementation first!

from linearclass.logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def compute_accuracy_positive(y_true, y_pred):
    true_pos = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            true_pos += 1
    return true_pos / np.sum(y_true == 1)


def compute_accuracy_negative(y_true, y_pred):
    true_neg = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            true_neg += 1
    return true_neg / np.sum(y_true == 0)

def compute_balanced_accuracy(y_true, y_pred):
    return (compute_accuracy_positive(y_true, y_pred) + compute_accuracy_negative(y_true, y_pred)) / 2

def main(train_path, validation_path, save_path):
    """Problem: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(validation_path, add_intercept=True)

    naive_model = LogisticRegression(verbose=True)
    naive_model.fit(x_train, y_train)

    y_val_naive = naive_model.predict(x_val)
    y_val_pred_naive = (y_val_naive >= 0.5).astype(int)

    A = np.mean(y_val_pred_naive == y_val)
    A0 = compute_accuracy_negative(y_val, y_val_pred_naive)
    A1 = compute_accuracy_positive(y_val, y_val_pred_naive)
    A_bal = compute_balanced_accuracy(y_val, y_val_pred_naive)
    print(f"[naive] A={A:.4f}, A_bal={A_bal:.4f}, A0={A0:.4f}, A1={A1:.4f}")

    plot_path = output_path_naive.replace('_pred.txt', '_plot.png')
    util.plot(x_val, y_val, naive_model.theta, plot_path)

    # Part (d): Upsampling minority class
    upsample_times = int(1 / kappa)
    minor = (y_train == 1)
    majority = (y_train == 0)

    x_minor = x_train[minor]
    y_minor = y_train[minor]
    x_majority = x_train[majority]
    y_majority = y_train[majority]
    
    x_minor_sampled = np.tile(x_minor, (upsample_times, 1))
    y_minor_sampled = np.tile(y_minor, upsample_times)

    x_train_up = np.concatenate([x_majority, x_minor_sampled])
    y_train_up = np.concatenate([y_majority, y_minor_sampled])

    print(x_train_up.shape, y_train_up.shape)

    upsampled_model = LogisticRegression(verbose=True)
    upsampled_model.fit(x_train_up, y_train_up)

    y_val_up = upsampled_model.predict(x_val)
    y_val_pred_up = (y_val_up >= 0.5).astype(int)

    A = np.mean(y_val_pred_up == y_val)
    A0 = compute_accuracy_negative(y_val, y_val_pred_up)
    A1 = compute_accuracy_positive(y_val, y_val_pred_up)
    A_bal = compute_balanced_accuracy(y_val, y_val_pred_up)
    print(f"[naive] A={A:.4f}, A_bal={A_bal:.4f}, A0={A0:.4f}, A1={A1:.4f}")

    plot_path = output_path_upsampling.replace('_pred.txt', '_plot.png')
    util.plot(x_val, y_val, upsampled_model.theta, plot_path)
    # *** END CODE HERE

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main(train_path=os.path.join(script_dir, 'train.csv'),
         validation_path=os.path.join(script_dir, 'validation.csv'),
         save_path=os.path.join(script_dir, 'imbalanced_X_pred.txt'))
