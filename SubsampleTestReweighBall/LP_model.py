# Very slow model.
# Need install gurobipy

class LP:
    pass
# import gurobipy as gp
# import numpy as np
# from gurobipy import GRB
#
# from misc import get_np
#
#
# class LP:
#     def __init__(self):
#         self.coef_ = None
#         self.intercept_ = None
#
#     def fit(self, X: np.ndarray, y: np.ndarray) -> None:
#         with gp.Env(empty=True) as env:  # No stdout
#             env.setParam('OutputFlag', 0)
#             env.start()
#             with gp.Model(env=env) as model:
#                 model.feasRelaxS(0, False, False, True)
#
#                 # Build model m here
#                 X = np.concatenate((X, get_np(np.ones((X.shape[0], 1)))), axis=1)
#                 sample_train_size = X.shape[0]
#                 sample_train_dim = X.shape[1]
#                 plus_minus_labels_train = y * 2 - 1  # convert {0,1} labels to {-1,1} labels
#                 sample_train_positive = X * plus_minus_labels_train.reshape(sample_train_size, 1)
#                 b = np.ones(sample_train_size)
#                 c = np.zeros(sample_train_dim)
#                 # Build optimization model
#                 # self.logger.info('\t\t\t\tStart building LP model')
#                 # model = gp.Model()
#                 w = model.addMVar(shape=(sample_train_dim,), lb=float('-inf'))
#                 model.addConstr(sample_train_positive @ w >= b)
#                 model.setObjective(expr=c @ w, sense=GRB.MINIMIZE)
#                 # model.write('mymodel.mps')
#                 # self.logger.info('\t\t\t\tStart to optimize LP model')
#                 model.optimize()
#                 # self.logger.info('\t\t\t\tFinish optimizing the LP model')
#                 w = np.array([v.x for v in model.getVars()])
#                 self.coef_ = w[:-1]
#                 self.intercept_ = w[-1]
#
#     def predict(self, X) -> np.ndarray:
#         return np.where((X @ self.coef_) + self.intercept_ > 0, 1, 0)
