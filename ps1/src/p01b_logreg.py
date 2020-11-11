import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def h(theta, x):
            """vectorized implementation of h_theta(x)
            Args:
                theta: Shape (n,)
                x: Shape (m,n)

            Returns:
                numpy array of shape (m,)
            """
            return 1 / (1 + np.exp(-np.dot(x, theta)))

        def gradient(x, y, theta):
            """
            Args:
                x: numpy array of shape (m,n)
                y: numpy array of shape (m,)
                theta: numpy array of shape (n,)

            """
            m, = y.shape
            return -1 / m * np.dot(x.T, y - h(theta, x))

        def hessian(x, y, theta):
            """
            Args:
                x: numpy array of shape (m,n)
                y: numpy array of shape (m,)
                theta: numpy array of shape (n,)

            """
            m, = y.shape
            htx = np.reshape(h(theta, x), (-1, 1))
            return 1 / m * np.dot(x.T, x * htx * (1 - htx))

        def nexttheta(theta):
            grad = gradient(x, y, theta)
            H = hessian(x, y, theta)
            H_inv = np.linalg.inv(H)
            return theta - np.dot(H_inv, grad)

        m, n = x.shape
        theta_prev = np.zeros(n)
        theta_next = nexttheta(theta_prev)

        while np.linalg.norm(theta_prev - theta_next, 1) > self.eps:
            theta_prev = theta_next
            theta_next = nexttheta(theta_prev)

        self.theta = theta_next
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (x @ self.theta) >= 0
        # *** END CODE HERE ***
