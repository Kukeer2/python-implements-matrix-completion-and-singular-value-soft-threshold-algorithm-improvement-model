import numpy as np
import pandas as pd
from numpy.linalg import svd


class MatrixCompletionSolver:
    def __init__(self):
        pass

    def singular_value_thresholding(self, Z, W, lambd):
        """
        奇异值阈值处理函数
        """
        # 对矩阵 Z - W / lambd 进行奇异值分解（SVD）
        U, S, VT = svd(Z - W / lambd, full_matrices=False)
        # 将奇异值进行软阈值处理，将小于 lambd / 2 的奇异值置零。
        S_thresh = np.maximum(S - lambd / 2, 0)
        # 用处理后的奇异值重新构建矩阵
        return U @ np.diag(S_thresh) @ VT

    # 随机缺失的处理
    def RM(self, missing_rate, dense_mat):
        dim = dense_mat.shape
        sparse_tensor = dense_mat * np.round(np.random.rand(*dim) + 0.5 - missing_rate)
        return sparse_tensor

    # 进行 L1 投影
    # L1 投影的目的是将输入矩阵的元素限制在一个 L1 范数小于等于某个给定值的子集内，以保证模型的稀疏性。
    def l1_projection(self, Z, epsilon):
        Z[Z < -epsilon] += epsilon
        Z[Z > epsilon] -= epsilon
        Z[(Z >= -epsilon) & (Z <= epsilon)] = 0
        return Z

    def admm_matrix_completion(self, dense_mat, A, gamma, lambd, maxiter, missing_rate, epsilon=1e-6):
        """
        ADMM矩阵补全算法
        """
        sparse_mat = self.RM(missing_rate, dense_mat)
        pos_train = np.where(sparse_mat != 0)
        pos_test = np.where((sparse_mat == 0) & (dense_mat != 0))
        X = sparse_mat.copy()
        Z = np.zeros(sparse_mat.shape)
        W = np.zeros(sparse_mat.shape)
        rse = np.zeros(maxiter)
        # 拉普拉斯矩阵（L=D-A)
        L = np.diag(A.sum(axis=0)) - A
        for it in range(maxiter):
            # X更新公式
            X = self.singular_value_thresholding(Z, W, 1 / lambd)
            X[pos_train] = sparse_mat[pos_train]

            # Z更新公式
            # Z =（γL.T*L+λI）-1 * (λX+W) + Y   (对应文中的公式)
            inv_term = np.linalg.inv(gamma * np.dot(L.T, L) + lambd * np.identity(L.shape[0]))
            Z_Omega_c = np.dot(inv_term, lambd * X + W)
            Z_Omega = sparse_mat
            Z = self.l1_projection(Z_Omega + Z_Omega_c, epsilon)

            # W更新公式
            W = W + lambd * (X - Z)
            rse[it] = (np.linalg.norm(X[pos_test] - dense_mat[pos_test], 2)
                       / np.linalg.norm(dense_mat[pos_test], 2))
            print(f"Iteration {it + 1}: rse = {rse[it]}")
        return X, rse

    def train_and_predict(self, adj_matrix_file, feature_matrix_file, gamma, lambd, maxiter, missing_rate):
        """
        训练和预测函数
        """
        # 读取测试数据
        adj_matrix = pd.read_csv(adj_matrix_file, header=None).values
        feature_matrix = pd.read_csv(feature_matrix_file, header=None).values

        # 构造观测张量Y
        dense_mat = feature_matrix.T

        # 执行ADMM算法
        result_matrix, rse = self.admm_matrix_completion(dense_mat, adj_matrix, gamma, lambd, maxiter, missing_rate)
        result_matrix = result_matrix.T
        return result_matrix, rse
