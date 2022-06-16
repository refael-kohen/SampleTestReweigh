import numpy as np
import pandas as pd

from SubsampleTestReweighBall.sgd_classifier_custom import SGDClassifierCustom

# Run example:
if __name__ == '__main__':
    random_state = None
    np.random.seed(random_state)
    sample_size = 10
    dim = 5
    # X = np.random.normal(size=(sample_size, dim))
    # y = np.random.normal(size=(sample_size,))
    # y = np.where(y < 0, 0, 1)

    from sklearn import datasets
    iris = datasets.load_iris(return_X_y=True, as_frame=True)
    iris = pd.concat((iris[0], iris[1]), axis=1)
    iris = iris[iris.loc[:,"target"]!=2].sample(frac=1)
    X = iris.iloc[:, 0:4]
    print(X)
    y = iris.iloc[:, 4]
    sample_size = X.shape[0]
    u_t = np.array([1/sample_size]*sample_size)
    # cls = SGDClassifierCustom(early_stopping=True, early_stopping_score=0.01, early_stopping_n_iter=50,
    #                           fit_intercept=True, alpha=1.0, random_state=random_state,
    #                           max_iter=300, verbose=2, loss='circle_hinge', eta0=0.1, private=False,
    #                           mean_noise=0.0, sigam_noise=50.0, batch_size=10, do_proj=True,
    #                           min_proj=0.0, max_proj=1.0, random_blocks_num=1, eta_start_iter=1, average=True)
    cls = SGDClassifierCustom(early_stopping=False, early_stopping_score=0.01, early_stopping_n_iter=50,
                              fit_intercept=True, alpha=1.0, random_state=random_state,
                              max_iter=5, verbose=2, loss='circle_hinge', eta0=0.1, private=True,
                              mean_noise=0.0, sigam_noise=50.0, batch_size=1, do_proj=True,
                              min_proj=0.0, max_proj=1.0, random_blocks_num=1, eta_start_iter=1, average=True)
    cls.fit(X, y, sample_dist=u_t, coef_init=[0.8, 0.45850706, 0.4, 0.4], intercept_init=0.1435852880613767)
    for log in cls.logs:
        if log:
            print(log)
    print('coef_: ', cls.coef_)
    print('intercept_: ', cls.intercept_)
    print('score: ', cls.score(X, y))
    print('predict: ', cls.predict(X).tolist())
    print('real  y: ', y.tolist())

