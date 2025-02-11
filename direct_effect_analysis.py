from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy
from sklearn.decomposition import PCA
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from mvlearn.embed import GCCA
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.covariance import ledoit_wolf, shrunk_covariance, OAS
from sklearn.decomposition import TruncatedSVD


def MANCOVA(X, Y, Z):
    data = {
        'X': X[:, 0],
        'Z': Z[:, 0]
    }

    formula = ''
    for i in range(Y.shape[1]):
        var = 'Y_' + str(i)
        data[var] = Y[:, i]
        formula += var + ' + '

    df = pd.DataFrame(data)
    formula = formula[:-2] + '~ X + Z'
    manova = MANOVA.from_formula(formula, data=df)
    result = manova.mv_test()
    return result

def fit_svd(X, Y, Z, center=False, scale=False, tol=None, rank=None, intercept=False, shrink=False):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    d = Y.shape[1]
    
    # Scale y if center or scale are specified
    if center or scale:
        scaler = StandardScaler(with_mean=center, with_std=scale)
        y_scaled = scaler.fit_transform(Y)
        cen = scaler.mean_
        sc = scaler.scale_
    else:
        y_scaled = Y
        cen = None
        sc = None

    # Add intercept if specified
    if intercept:
        Z = np.hstack([Z, np.ones((Z.shape[0], 1))])

    zx = np.hstack([Z, X])

    if zx.shape[1] < zx.shape[0]:
        # QR decomposition
        Q, R = np.linalg.qr(zx)
        Q2 = Q[:, Z.shape[1]:]
    else:
        raise ValueError('insufficient observations, try reducing dimensionality of x and/or z.')

    Y2 = np.dot(Q2.T, Y)

    n, p = Y2.shape

    if rank is not None:
        assert isinstance(rank, int) and rank > 0, "rank_ must be a positive integer."
        k = min(rank, n, p)
    else:
        k = min(n, p)

    # Perform SVD
    U, u, W = np.linalg.svd(Y2, full_matrices=False)
    return u, W[:k].T  # Take the first k columns of W


def fit_chow(X, Y, Z, reg_full, reg_res, solver='T_F', alpha=1e-5, shrink=False):
    n = X.shape[0]
    reg_full.fit(np.hstack((X, Z)), Y)
    reg_res.fit(Z, Y)
    RSS_full = Y - reg_full.predict(np.hstack((X, Z)))
    RSS_res = Y - reg_res.predict(Z)
    N = RSS_full.T @ RSS_full
    M = N  - RSS_res.T @ RSS_res

    if solver == 'T_F':
        u, W = scipy.linalg.eigh(M, N + alpha*np.identity(N.shape[0]))
    elif solver == 'T_S':
        u, W = np.linalg.eigh(M)
    # idx = np.argsort(u)[::-1]
    # u, W = u[idx], W[:, idx]
    return u, W

def fit_optimal_detector(X, Y, Z, reg_full, reg_res, alpha=1e-5, shrink=False):
    n = X.shape[0]
    p, d, r = X.shape[1], Y.shape[1], Z.shape[1]
    reg_full.fit(np.hstack((X, Z)), Y)
    reg_res.fit(Z, Y)
    RSS_full = Y - reg_full.predict(np.hstack((X, Z)))
    RSS_dn = Y - reg_full.predict(np.hstack((X, np.zeros(Z.shape))))
    RSS_res = Y - reg_res.predict(Z)
    if shrink:
        svd = TruncatedSVD(k=50)
        N = svd.fit_transform(np.cov(RSS_full))
        M = N  - svd.fit_transform(np.cov(RSS_res))
    else : 
        N = RSS_dn.T @ RSS_dn
        M = RSS_full.T @ RSS_full  - RSS_res.T @ RSS_res

        # M2 = RSS_res.T @ RSS_res


    u, W = scipy.linalg.eigh(M, N + alpha*np.identity(N.shape[0]))
    # u2, W2 = scipy.linalg.eigh(M2, N + alpha*np.identity(N.shape[0]))

    return u, W, reg_full

def fit_GCM(X, Y, Z, reg_X, reg_Y, mva):
    reg_X.fit(Z, X)
    reg_Y.fit(Z, Y)
    RSS_X = X - reg_X.predict(Z)
    RSS_Y = Y - reg_Y.predict(Z)

    mva = CCA(n_components=1)
    mva.fit(RSS_X, RSS_Y)
    return mva

def fit_CCA(X, Y, Z, reg_X, reg_Y, alpha=1e-6):
    reg_X.fit(Z, X)
    reg_Y.fit(Z, Y)
    RSS_X = X - reg_X.predict(Z)
    RSS_Y = Y - reg_Y.predict(Z)

    Y_mean = RSS_Y.mean(axis=0)
    RSS_X = RSS_X - RSS_X.mean(axis=0)
    RSS_Y = RSS_Y - Y_mean

    # Step 2: Compute covariance matrices
    Cxx = RSS_X.T @ RSS_X + alpha * np.eye(X.shape[1])
    Cyy = RSS_Y.T @ RSS_Y + alpha * np.eye(Y.shape[1])
    Cxy = RSS_X.T @ RSS_Y
    Cyx = Cxy.T
    
    # Step 3: Solve the generalized eigenvalue problem
    # Solve for eigenvalues and eigenvectors of (Cxx^-1 * Cxy * Cyy^-1 * Cyx)
    u, W = scipy.linalg.eig(Cyx @ np.linalg.pinv(Cxx) @ Cxy, Cyy )

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(u)[::-1]
    u = u[sorted_indices]
    W = W[:, sorted_indices]

    return u.real, W.real, reg_Y, Y_mean

def fit_gcca(X, Y, Z):
    mva = GCCA()
    mva.fit([X, Z, Y])
    return mva


class DirectEffectAnalysis:
    def __init__(self, type ='T_D', regressor_0=LinearRegression(), regressor_1=LinearRegression(), alpha=1e-5, shrink=False):
        self.regressor_0 = regressor_0
        self.regressor_1 = regressor_1
        self.type = type
        self.alpha = alpha
        self.shrink = shrink

    def fit(self, X, Y, Z):
        if self.type == 'T_F' or self.type == 'T_S':
            self.u, self.W = fit_chow(X, Y, Z, self.regressor_0, self.regressor_1, self.type, self.alpha, shrink=self.shrink)
        elif self.type == 'PLS' :
            self.mva = fit_GCM(X, Y, Z, self.regressor_0, self.regressor_1, PLSCanonical())
        elif self.type == 'pCCA' :
            self.u, self.W, self.reg_Y, self.Y_mean = fit_CCA(X, Y, Z, self.regressor_0, self.regressor_1, self.alpha)
        elif self.type == 'GCM' :
            self.mva = fit_GCM(X, Y, Z, self.regressor_0, self.regressor_1, CCA())
        elif self.type == 'GCCA' :
            self.mva = fit_gcca(X, Y, Z)
        elif self.type == 'T_S':
            self.u, self.W = fit_svd(X, Y, Z)
        elif self.type == 'PCA':
            self.pca = PCA(n_components=1)
            self.pca.fit(Y)
        elif self.type == 'T_D':
            self.u, self.W, self.reg_full = fit_optimal_detector(X, Y, Z, self.regressor_0, self.regressor_1, self.alpha, shrink=self.shrink)


    def transform(self, X, Y, Z):
        if self.type == 'T_F' or self.type == 'T_S' or self.type == 'T_D' :
            return Y @ self.W[:, 0]
        elif self.type == 'pCCA':
            RSS_Y = Y - self.reg_Y.predict(Z)
            RSS_Y = RSS_Y - self.Y_mean
            return RSS_Y @ self.W[:, 0]
        elif self.type == 'PLS' or self.type == 'GCM' :
            return self.mva.transform(X, Y)[1][:, 0]
        elif self.type == 'PCA':
            return self.pca.transform(Y)[:, 0]
        elif self.type == 'GCCA':
            return self.mva.transform([X, Y, Z])[1, :, 0]
        
    def fit_transform(self, X, Y, Z):
        self.fit(X, Y, Z)
        return self.transform(X, Y, Z)

    def test(self, X, Y, Z):
        if self.type == 'OptiDet':
            w = self.W[:, 0]
            N = self.reg_full.predict(np.hstack((X, np.zeros(Z.shape))))
            sigma = (w.T @ np.cov(N.T) @ w)
            T = (Y @ w) / (sigma**0.5)
            p_value = scipy.stats.norm.cdf(T, 0, 1)
            return np.min(p_value), T
        elif self.type == 'CCA':
            rho = np.sqrt(self.u[0])

 