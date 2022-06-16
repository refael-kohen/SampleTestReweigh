import numpy as np
# noinspection PyUnresolvedReferences
from sklearn.linear_model._stochastic_gradient import _prepare_fit_binary, BaseSGDClassifier, SGDClassifier, \
    check_random_state, make_dataset, MAX_INT, DEFAULT_EPSILON

# from SubsampleTestReweighCycle.private_svm_sgd._sgd_fast_custom import EpsilonInsensitive
# from SubsampleTestReweighCycle.private_svm_sgd._sgd_fast_custom import Hinge
# from SubsampleTestReweighCycle.private_svm_sgd._sgd_fast_custom import Huber
# from SubsampleTestReweighCycle.private_svm_sgd._sgd_fast_custom import Log
# from SubsampleTestReweighCycle.private_svm_sgd._sgd_fast_custom import ModifiedHuber
# from SubsampleTestReweighCycle.private_svm_sgd._sgd_fast_custom import SquaredEpsilonInsensitive
# from SubsampleTestReweighCycle.private_svm_sgd._sgd_fast_custom import SquaredHinge
# from SubsampleTestReweighCycle.private_svm_sgd._sgd_fast_custom import SquaredLoss
from SubsampleTestReweighBall.private_svm_sgd._sgd_fast_custom import CircleHinge
from SubsampleTestReweighBall.private_svm_sgd._sgd_fast_custom import _plain_sgd


# NOISE_CHANGES
# This is copy of the scikit-learn function from:
# sklearn\linear_model\_stochastic_gradient.py
# with addition of parameters. 
# This function calls (with the additional parameters) to _plain_sgd function (implemented in _plain_sgd.pyx cython file).
def fit_binary(
        est,
        i,
        X,
        y,
        alpha,
        C,
        learning_rate,
        max_iter,
        pos_weight,
        neg_weight,
        sample_weight,
        validation_mask=None,
        random_state=None,
        # NOISE_CHANGES - New parameters
        random_blocks_num=1,
        eta_start_iter=1,
        logs=None,
        early_stopping_score=0.0,
        early_stopping_n_iter=100,
        sample_dist=None,
        private=None,
        mean_noise=None,
        sigma_noise=None,
        batch_size=1,
        do_proj=1,
        min_proj=0,
        max_proj=1,
):
    """Fit a single binary classifier.

    The i'th class is considered the "positive" class.

    Parameters
    ----------
    est : Estimator object
        The estimator to fit

    i : int
        Index of the positive class

    X : numpy array or sparse matrix of shape [n_samples,n_features]
        Training data

    y : numpy array of shape [n_samples, ]
        Target values

    alpha : float
        The regularization parameter

    C : float
        Maximum step size for passive aggressive

    learning_rate : str
        The learning rate. Accepted values are 'constant', 'optimal',
        'invscaling', 'pa1' and 'pa2'.

    max_iter : int
        The maximum number of iterations (epochs)

    pos_weight : float
        The weight of the positive class

    neg_weight : float
        The weight of the negative class

    sample_weight : numpy array of shape [n_samples, ]
        The weight of each sample

    validation_mask : numpy array of shape [n_samples, ], default=None
        Precomputed validation mask in case _fit_binary is called in the
        context of a one-vs-rest reduction.

    random_state : int, RandomState instance, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    # NOISE_CHANGES ####
    # noise_seed is the seed of gaussian noise. if noise_seed==0 it is depends on the time,
    # otherwise it is equals to random_state
    private = 1 if private else 0
    if random_state is None:
        noise_seed = 0
    else:
        noise_seed = random_state
    ##### until now  - NOISE_CHANGES

    # if average is not true, average_coef, and average_intercept will be
    # unused
    y_i, coef, intercept, average_coef, average_intercept = _prepare_fit_binary(
        est, y, i
    )
    assert y_i.shape[0] == y.shape[0] == sample_weight.shape[0]

    random_state = check_random_state(random_state)
    dataset, intercept_decay = make_dataset(
        X, y_i, sample_weight, random_state=random_state
    )

    penalty_type = est._get_penalty_type(est.penalty)
    learning_rate_type = est._get_learning_rate_type(learning_rate)

    if validation_mask is None:
        validation_mask = est._make_validation_split(y_i)
    classes = np.array([-1, 1], dtype=y_i.dtype)
    # NOISE_CHANGES - Remove the validation of the parameters
    # validation_score_cb = est._make_validation_score_cb(
    #     validation_mask, X, y_i, sample_weight, classes=classes
    # )

    # numpy mtrand expects a C long which is a signed 32 bit integer under
    # Windows
    seed = random_state.randint(MAX_INT)

    tol = est.tol if est.tol is not None else -np.inf
    coef, intercept, average_coef, average_intercept, n_iter_ = _plain_sgd(
        coef,
        intercept,
        est.average,
        average_coef,
        average_intercept,
        est.loss_function_,
        alpha,
        dataset,
        int(max_iter),
        int(est.fit_intercept),
        seed,
        est.eta0,
        # NOISE_CHANGES
        random_blocks_num,
        eta_start_iter,
        logs,
        est.early_stopping,
        float(early_stopping_score),
        int(early_stopping_n_iter),
        sample_dist,
        batch_size,
        do_proj,
        min_proj,
        max_proj,
        private,
        mean_noise,
        sigma_noise,
        noise_seed,
        int(est.verbose),
    )

    if est.average:
        if len(est.classes_) == 2:
            est._average_intercept[0] = average_intercept
        else:
            est._average_intercept[i] = average_intercept

    return coef, intercept, n_iter_


# NOISE_CHANGES
# SGDClassifier inharits the BaseSGDClassifier from:
# sklearn\linear_model\_stochastic_gradient.py
# Here we override the functions of BaseSGDClassifier
# No changes made. the function _fit_binary call to local function fit_binary 
# (in this file) instead of fit_binary of scikit-learn.
class SGDClassifierCustom(SGDClassifier):
    def __init__(self, early_stopping_score=0.0, early_stopping_n_iter=100, private=True, mean_noise=0, sigam_noise=0,
                 batch_size=1, do_proj=1, min_proj=0, max_proj=1, random_blocks_num=1, eta_start_iter=1, *args, **kwargs):
        super(SGDClassifierCustom, self).__init__(*args, **kwargs)
        # NOISE_CHANGES - adding the folloing parameters
        # self.early_stopping=early_stopping, #already exists in original class
        self.logs = np.array(['' * 1000 for _ in range(100)], dtype=object)
        self.early_stopping_score = early_stopping_score
        self.early_stopping_n_iter = early_stopping_n_iter
        self.private = private
        self.mean_noise = mean_noise
        self.sigma_noise = sigam_noise
        self.batch_size = batch_size
        self.do_proj = do_proj
        self.min_proj = min_proj
        self.max_proj = max_proj
        self.random_blocks_num = random_blocks_num
        self.eta_start_iter = eta_start_iter

    # TODO: Remove squared_loss in v1.2
    loss_functions = {
        "circle_hinge": (CircleHinge, 0.0),
        # "hinge": (Hinge, 1.0),
        # "squared_hinge": (SquaredHinge, 1.0),
        # "perceptron": (Hinge, 0.0),
        # "log": (Log,),
        # "modified_huber": (ModifiedHuber,),
        # "squared_error": (SquaredLoss,),
        # "squared_loss": (SquaredLoss,),
        # "huber": (Huber, DEFAULT_EPSILON),
        # "epsilon_insensitive": (EpsilonInsensitive, DEFAULT_EPSILON),
        # "squared_epsilon_insensitive": (SquaredEpsilonInsensitive, DEFAULT_EPSILON),
    }

    # NOISE_CHANGES
    def fit(self, *args, **kwargs):
        if 'sample_dist' in kwargs.keys():
            self.sample_dist = kwargs['sample_dist']
            del kwargs['sample_dist']
        super(SGDClassifierCustom, self).fit(*args, **kwargs)

    def _fit_binary(self, X, y, alpha, C, sample_weight, learning_rate, max_iter):
        """Fit a binary classifier on X and y."""
        coef, intercept, n_iter_ = fit_binary(
            self,
            1,
            X,
            y,
            alpha,
            C,
            learning_rate,
            max_iter,
            self._expanded_class_weight[1],
            self._expanded_class_weight[0],
            sample_weight,
            random_state=self.random_state,

            # NOISE_CHANGES - adding parameters
            random_blocks_num=self.random_blocks_num,
            eta_start_iter=self.eta_start_iter,
            logs=self.logs,
            early_stopping_score=self.early_stopping_score,
            early_stopping_n_iter=self.early_stopping_n_iter,
            sample_dist=self.sample_dist,
            private=self.private,
            mean_noise=self.mean_noise,
            sigma_noise=self.sigma_noise,
            batch_size=self.batch_size,
            do_proj=self.do_proj,
            min_proj=self.min_proj,
            max_proj=self.max_proj,
        )

        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_

        # need to be 2d
        if self.average > 0:
            if self.average <= self.t_ - 1:
                self.coef_ = self._average_coef.reshape(1, -1)
                self.intercept_ = self._average_intercept
            else:
                self.coef_ = self._standard_coef.reshape(1, -1)
                self._standard_intercept = np.atleast_1d(intercept)
                self.intercept_ = self._standard_intercept
        else:
            self.coef_ = coef.reshape(1, -1)
            # intercept is a float, need to convert it to an array of length 1
            self.intercept_ = np.atleast_1d(intercept)


# NOISE_CHANGES
# Run example:
# you need to move this script out of private_svd_sgd folder,
# or change the "from ._sgd_fast import ..." to "from _sgd_fast import ..." (without dot) in the begin of this file.
if __name__ == '__main__':
    random_state = None
    np.random.seed(random_state)
    X = np.random.normal(size=(80000, 500))
    y = np.random.normal(size=(80000,))
    y = np.where(y < 0, 0, 1)
    u_t = np.array([1 / 80000] * 80000)
    cls = SGDClassifierCustom(early_stopping=True, early_stopping_score=0.01, early_stopping_n_iter=100,
                              fit_intercept=True, alpha=0.0000000000001, random_state=random_state,
                              max_iter=1000, verbose=10, loss='circle_hinge', eta0=0.1, private=True,
                              mean_noise=0, sigam_noise=2, batch_size=1, do_proj=1,
                              min_proj=0, max_proj=1, random_blocks_num=1, eta_start_iter=1, average=False)
    cls.fit(X, y, sample_dist=u_t)
    print('coef_: ', cls.coef_)
    cls.score(X, y)
