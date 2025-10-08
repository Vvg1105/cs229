import numpy as np
import util
import os

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_valid, y_valid, model.theta, plot_path)
    
    y_pred_valid = model.predict(x_valid)
    np.savetxt(save_path, y_pred_valid)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        
        for i in range(self.max_iter):

            n = x.shape[0]

            h = self.sigmoid(x @ self.theta)

            if self.verbose:
                epsilon = 1e-5
                h_clipped = np.clip(h, epsilon, 1 - epsilon)
                loss = -(y * np.log(h_clipped) + (1 - y) * np.log(1 - h_clipped))
                avg_loss = np.mean(loss)
                print(f"loss={avg_loss:.6f} for iteration {i}")

            J = (1/n) * x.T @ (h - y)

            sig_h_grad = h * (1 - h)

            sig_h_grad_matrix = np.diag(sig_h_grad)

            H = (1/n) * x.T @ sig_h_grad_matrix @ x

            theta_new = self.theta - self.step_size * (np.linalg.inv(H) @ J)

            if np.linalg.norm(theta_new - self.theta) < self.eps:
                self.theta = theta_new
                break

            self.theta = theta_new
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self.sigmoid(x @ self.theta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    main(train_path=os.path.join(script_dir, 'ds1_train.csv'),
         valid_path=os.path.join(script_dir, 'ds1_valid.csv'),
         save_path=os.path.join(script_dir, 'logreg_pred_1.txt'))

    main(train_path=os.path.join(script_dir, 'ds2_train.csv'),
         valid_path=os.path.join(script_dir, 'ds2_valid.csv'),
         save_path=os.path.join(script_dir, 'logreg_pred_2.txt'))