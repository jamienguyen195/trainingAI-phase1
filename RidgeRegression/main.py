import numpy as np
import math

with open("deathRate.txt") as f:
    content = f.read()

n_col = 17
x = content.split()
clean_data = []
temp_list = []
for i in range(len(x)):
    if i % n_col == n_col - 1:
        temp_list.append(x[i])
        clean_data.append(temp_list)
        temp_list = []
    else:
        temp_list.append(x[i])

table = np.asarray(clean_data, dtype="float")


def normalize_and_add_ones(arr):
    x_max = np.array([[np.amax(arr[:, col]) for col in range(arr.shape[1])] for _ in range(arr.shape[0])])
    x_min = np.array([[np.amin(arr[:, col]) for col in range(arr.shape[1])] for _ in range(arr.shape[0])])

    x_normalized = (arr - x_min) / (x_max - x_min)
    ones = np.ones((arr.shape[0], 1))
    return np.column_stack((ones, x_normalized))


Y = table[:, -1:]
table = table[:, 1:-1]
X = normalize_and_add_ones(table)


class RidgeRegression:
    def __init__(self):
        return

    def fit(self, X_train, Y_train, LAMBDA):
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]
        W = np.linalg.inv(X_train.transpose().dot(X_train) + LAMBDA * np.identity(X_train.shape[1])).dot(
            X_train.transpose()).dot(Y_train)
        return W

    def fit_gradient(self, X_train, Y_train, LAMBDA, learning_rate, max_epoch=100, batch_size=128):
        W = np.random.rand(X_train.shape[1])
        last_loss = math.inf
        for ep in range(max_epoch):
            arr = np.array(range(X_train[0]))
            np.random.shuffle(arr)
            X_train = X_train[arr]
            Y_train = Y_train[arr]
            num_batch = int(np.ceil(X_train.shape[0] / batch_size))
            for i in range(num_batch):
                index = i * batch_size
                X_train_sub = X_train[index:index + batch_size]
                Y_train_sub = Y_train[index:index + batch_size]
                grad = X_train_sub.T.dot(X_train_sub.dot(W) - Y_train_sub) + LAMBDA * W
                W = W - learning_rate * grad
            new_loss = self.compute_RSS(Y_train, self.predict(W, X_train))
            if np.abs(new_loss - last_loss) < 1e-5:
                break
            last_loss = new_loss
        return W

    def predict(self, W, X_new):
        Y_predict = X_new.dot(W)
        return Y_predict

    def compute_RSS(self, Y_new, Y_predict):
        loss = 1 / Y_new.shape[0] * sum((Y_new - Y_predict) ** 2)
        return loss

    def get_the_best_LAMBDA(self, X_train, Y_train, scan_range=50):
        def cross_validation(num_folds, LAMBDA):
            RSS = 0
            pop_fold = X_train.shape[0] // num_folds + 1
            for i in range(num_folds):
                valid_part = {"X": X_train[i * pop_fold:min((i + 1) * pop_fold, X_train.shape[0]), :],
                              "Y": Y_train[i * pop_fold:min((i + 1) * pop_fold, Y_train.shape[0]), :]}
                train_part = {"X": np.row_stack(
                    (X_train[:i * pop_fold, :], X_train[min((i + 1) * pop_fold, X_train.shape[0]):, :])),
                              "Y": np.row_stack(
                                  (Y_train[:i * pop_fold, :], Y_train[min((i + 1) * pop_fold, Y_train.shape[0]):, :]))}
                W = self.fit(train_part["X"], train_part["Y"], LAMBDA)
                Y_predict = self.predict(W, valid_part["X"])
                RSS += self.compute_RSS(valid_part["Y"], Y_predict)
            return RSS / num_folds

        min_RSS = math.inf
        best_LAMBDA = 0
        LAMBDA_values = [i / 1000 for i in range(scan_range * 1000)]
        for LAMBDA in LAMBDA_values:
            if cross_validation(5, LAMBDA) < min_RSS:
                best_LAMBDA = LAMBDA
                min_RSS = cross_validation(5, LAMBDA)
        return best_LAMBDA


X_train, Y_train = X[:50], Y[:50]
X_test, Y_test = X[50:], Y[50:]
ridge_regression = RidgeRegression()
best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, Y_train)
print("Best LAMBDA: ", best_LAMBDA)

W_learned = ridge_regression.fit(X_train, Y_train, best_LAMBDA)
Y_predict = ridge_regression.predict(W_learned, X_test)
RSS = ridge_regression.compute_RSS(Y_test, Y_predict)
print("RSS: ", RSS)
