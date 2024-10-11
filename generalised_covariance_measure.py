import numpy as np
from scipy import stats
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, LinearRegression

"""
This is an implementation for Generalized Covariance Measure in python.
Implemented by Xueda Shen. shenxueda@berkeley.edu
Rajen D. Shah, Jonas Peters: 
"The Hardness of Conditional Independence Testing and the Generalised Covariance Measure"
Annals of Statistics 48(3), 1514--1538, 2020.

"""


def GCM_translation(X, Y, Z, alpha = 0.05, nsim = 1000, res_X = None, res_Y = None, regressor=LinearRegression()):
    """
    This is a direct translation from GCM Testing in R. 
    
    Inputs:
        X, Y, Z: iid data for testing conditional independence.
        alpha: Pre-specified level of cutoff.
        nsim: Number of monte-carlo simulations.
        res_X, res_Y: corresponding to resid.XonZ, resid.YonZ respectively in original implementation. 
    """

    # Checking if providing residual or computing residual
    if Z is None:
        # Is residual provided
        if res_X is None:
            res_X = X
        if res_Y is None:
            res_Y = Y
    else:
        if res_X is None:
            if X is None:
                raise ValueError('Either X on residual of X given Z has to be provided.')
            if len(X.shape) == 1 or X.shape[1] == 1:
                if len(Z.shape) == 1:
                    Z = np.expand_dims(Z, axis = 1)
                # Compute residuals of X|Z. I chose Lasso as estimator. 
                model_X = regressor.fit(Z, X)
                res_X = X - model_X.predict(Z)
            else:
                model_X = regressor.fit(Z, X)
                res_X = X - model_X.predict(Z)
    if Y is None:
        raise ValueError ('Either Y on residual of Y given Z has to be provided.')
    if len(Y.shape) == 1 or Y.shape[1] == 1:
        model_Y = regressor.fit(Z, Y)
        res_Y = Y - model_Y.predict(Z)
    else:
        model_Y = regressor.fit(Z, Y)
        res_Y = Y - model_Y.predict(Z)

    if (len(res_X.shape) > 1 or len(res_Y.shape) > 1):
        # Obtaining covariance and test statistics
        d_X = res_X.shape[1]; d_Y = res_Y.shape[1]; nn = res_X.shape[0]

        # rep(times) in R is really np.tile. 
        # Translating R_mat = rep(....) * as.numeric(...)[, rep(....)]
        left = np.tile(res_X, reps = d_Y)  # rep(resid.XonZ, times = d_Y)
        left = left.flatten('F')
        right = res_Y[:, np.tile(np.arange(d_Y), reps = d_X)].flatten(order = 'F')   # as.numeric(as.matrix(resid.YonZ)[, rep(seq_len(d_Y), each=d_X)])
        R_mat = np.multiply(left, right)
        R_mat = R_mat.flatten(order = 'F')
        R_mat = np.reshape(R_mat, (nn, d_X * d_Y), order = 'F')
        R_mat = np.transpose(R_mat)

        norm_con = np.sqrt(np.mean(R_mat ** 2, axis = 1) - np.mean(R_mat, axis = 1)**2)
        norm_con = np.expand_dims(norm_con, axis = 1)
        R_mat = R_mat / norm_con

        # Test statistics
        test_stat = np.max(np.abs(np.mean(R_mat, axis = 1))) * np.sqrt(nn)
        noise = np.random.randn(nn, nsim)
        test_stat_sim = np.abs(R_mat @ noise)
        test_stat_sim = np.amax(test_stat_sim, axis = 0) / np.sqrt(nn)

        # p value
        pval = (np.sum(test_stat_sim >= test_stat) + 1) / (nsim + 1)
    
    else:
        if (len(res_X.shape) == 1):  # ifesle(is.null()...)
            nn = res_X.shape[0]
        else:
            nn = res_X.shape[0]
        R = np.multiply(res_X, res_Y)
        R_sq = R ** 2
        meanR = np.mean(R)
        test_stat = np.sqrt(nn) * meanR / np.sqrt(np.mean(R_sq) - meanR ** 2)
        pval = 2 * (1 - stats.norm.cdf(np.abs(test_stat)))


    return test_stat, pval