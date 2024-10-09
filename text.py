import numpy as np
import pandas as pd

def LRMC(sparse_mat, dense_mat, alpha, rho, maxiter):
    pos_train = np.where(sparse_mat != 0)
    pos_test = np.where((sparse_mat == 0) & (dense_mat != 0))
    binary_mat = sparse_mat.copy()
    binary_mat[pos_train] = 1
    print("pos_test", pos_test)

    X = sparse_mat.copy()
    Z = np.zeros(sparse_mat.shape)
    T = np.zeros(sparse_mat.shape)
    rse = np.zeros(maxiter)

    for it in range(maxiter):
        u, s, v = np.linalg.svd(X + T / rho, full_matrices=False)
        vec = s - alpha / rho
        vec[np.where(vec < 0)] = 0
        Z = np.matmul(np.matmul(u, np.diag(vec)), v)
        X = Z - T / rho
        X[pos_train] = sparse_mat[pos_train]
        T = T - rho * (Z - X)
        rse[it] = (np.linalg.norm(X[pos_test] - dense_mat[pos_test], 2)
                   / np.linalg.norm(dense_mat[pos_test], 2))
        print(f"Iteration {it + 1}: rse = {rse[it]}")
    return X, rse


def RM(missing_rate, dense_mat):
    dim = dense_mat.shape
    sparse_tensor = dense_mat * np.round(np.random.rand(*dim) + 0.5 - missing_rate)
    return sparse_tensor


# 示例用法
dense_mat = pd.read_csv('sz_speed.csv', header=None).values  # 假设原始矩阵是一个随机矩阵
missing_rate = 0.2
sparse_mat = RM(missing_rate, dense_mat)
alpha = 0.1
rho = 0.005
maxiter = 200
mat_hat, rse_lr = LRMC(sparse_mat, dense_mat, alpha, rho, maxiter)

# 打印MAE结果
print("X", mat_hat)
print("Final MAE for completed values:", rse_lr[-1])
result_df = pd.DataFrame(mat_hat)

result_df.to_csv('result_matrix.csv', index=False)

print("Result matrix saved to result_matrix.csv")