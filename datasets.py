from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np


class Bandit_multi:
    def __init__(self, name, is_shuffle=True, seed=None):
        # Fetch data
        if name == 'mushroom':
            X, y = fetch_openml('mushroom', version=1, return_X_y=True, as_frame=False)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'shuttle':
            X, y = fetch_openml('shuttle', version=1, return_X_y=True, as_frame=False)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        else:
            raise RuntimeError('Dataset does not exist')
        # Shuffle data
        if is_shuffle:
            self.X, self.y = shuffle(X, y, random_state=seed)
        else:
            self.X, self.y = X, y
        # generate one_hot coding:
        self.y_arm = OrdinalEncoder(
            dtype=np.int).fit_transform(self.y.reshape((-1, 1)))
        # cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = np.max(self.y_arm) + 1
        self.dim = self.X.shape[1] * self.n_arm
        self.act_dim = self.X.shape[1]

    def step(self):
        assert self.cursor < self.size
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                self.act_dim] = self.X[self.cursor]
        arm = self.y_arm[self.cursor][0]
        rwd = np.zeros((self.n_arm,))
        rwd[arm] = 1
        self.cursor += 1
        return X, rwd

    def finish(self):
        return self.cursor == self.size

    def reset(self):
        self.cursor = 0


class SyntheticDataset():
    def __init__(self, n_features: int, n_arms: int, order: int = 3, noise_std: float = 0.1, T: int = 6000, seed: int = None) -> None:
        self.n_features = n_features
        self.n_arms = n_arms
        self.order = order
        self.rng = np.random.default_rng(seed)
        # Generate random function parameters
        self.alphas = self.rng.uniform(low=-1, high=1, size=(n_arms, n_features * order + 1))
        # Normalise weights so every function is on the same scale
        alpha_norm = np.linalg.norm(self.alphas, axis=1, ord=2, keepdims=True)
        self.alphas /=  alpha_norm
        self.noise_std = noise_std
        self.T = T

        self.X, self.y = self.generate_dataset(self.T, (-1, 1))
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = self.y.shape[1]
        self.dim = self.X.shape[1] * self.n_arm
        self.act_dim = self.X.shape[1]


    def step(self):
        assert self.cursor < self.size
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                self.act_dim] = self.X[self.cursor]
        rwd = self.y[self.cursor]
        self.cursor += 1
        return X, rwd


    def generate_dataset(self, T: int, space: tuple):
        """
        Generate T samples for each arm based on the synthetic dataset configuration. 
        """

        X = np.array([self.rng.uniform(low=space[0], high=space[1], size=self.n_features) for _ in range(T)])
        y = np.array([self._generator(x) for x in X])

        X = normalize(X)

        return X, y

        
    def _generator(self, x: np.array) -> np.array:
        """
        Generate and return a reward for each arm 
        based on the input context.
        """
        x_temp = np.array([1])
        for i in range(1, self.order + 1) :
            x_temp = np.append(x_temp, x**i)
        f = np.matmul(self.alphas, x_temp.T)

        return f


if __name__ == '__main__':
    b = Bandit_multi('mushroom')
    x_y, a = b.step()
    print(x_y)
    print(np.linalg.norm(x_y[0]))
    print(x_y.shape)
    print(b.dim)