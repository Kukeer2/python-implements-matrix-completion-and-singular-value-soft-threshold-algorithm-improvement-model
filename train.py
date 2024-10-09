import pandas as pd
from ADMM import MatrixCompletionSolver

solver = MatrixCompletionSolver()
result_matrix, rse = solver.train_and_predict(adj_matrix_file='sz_adj.csv',
                                              feature_matrix_file='sz_speed.csv',
                                              gamma=0.01,
                                              lambd=0.001,
                                              maxiter=200,
                                              missing_rate=0.2)
result_df = pd.DataFrame(result_matrix)
print("Final MAE for completed values:", rse[-1])
result_df.to_csv('result_matrix.csv', index=False)

