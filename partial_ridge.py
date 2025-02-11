import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import scipy

class PartialRidgeCV:
    """
    Partial Ridge Regression with Cross-Validation.

    This class performs Ridge Regression with a custom regularization approach,
    using eigenvalue decomposition (EVD) for matrix inversion. The best 
    regularization parameter (`alpha`) is selected using cross-validation.

    Parameters
    ----------
    alphas : list or array-like
        Array of regularization parameters to test.
    reg_idx : int
        Index for selecting a specific response variable in multi-response settings.
    n_cv : int
        Number of folds for cross-validation.
    scoring : callable, optional
        Scoring function to evaluate model performance. Must accept `y_true` and
        `y_pred` as inputs and return a scalar. Default is `r2_score`.

    Attributes
    ----------
    best_alpha_ : float
        The regularization parameter with the best mean score.
    best_score_ : float
        The best mean score across all folds.
    coef_ : ndarray of shape (n_features,) or (n_features, n_responses)
        Coefficients of the fitted model.
    """
    def __init__(self, alphas, reg_idx, n_cv, scoring='r2'):
        self.alphas = alphas
        self.reg_idx = reg_idx
        self.n_cv = n_cv
        self.scoring = scoring
        self.scoring_functions = {
            'r2': r2_score,
            'mse': mean_squared_error,
        }

    def fit(self, X, y):
        """
        Fit the model using cross-validation and select the best alpha.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix.
        y : ndarray of shape (n_samples,) or (n_samples, n_responses)
            Target vector or matrix.

        Returns
        -------
        self : PartialRidgeCV
            Fitted instance of the model.
        """

        # Determine the scoring function to use
        if isinstance(self.scoring, str):
            if self.scoring not in self.scoring_functions:
                raise ValueError(f"Unsupported scoring method '{self.scoring}'. Supported: {list(self.scoring_functions.keys())}")
            scoring_function = self.scoring_functions[self.scoring]
        elif callable(self.scoring):
            scoring_function = self.scoring
        else:
            raise ValueError("Scoring must be a string or a callable.")
        
        # Initialize KFold for cross-validation
        kf = KFold(n_splits=self.n_cv, shuffle=True, random_state=0)
        
        # To track scores for each alpha
        alpha_scores = {alpha: [] for alpha in self.alphas}
        # Cross-validation loop
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Compute EVD on the training data
            XtX = X_train.T @ X_train
            XtY = X_train.T @ y_train
            u, Q = scipy.linalg.eigh(XtX)
            Y_prime = Q.T @ XtY

            # Compute coefficients for each alpha
            for alpha in self.alphas:
                d = 1 / (u + alpha*self.reg_idx)
                D = np.diag(d)
                beta = Q @ D @ Y_prime
                
                # Predict on the test set
                y_pred = X_test @ beta
                
                # Compute and store the score
                if self.scoring == 'mse':
                    score = -scoring_function(y_test, y_pred)  # Negate MSE for compatibility
                else:
                    score = scoring_function(y_test, y_pred)
                alpha_scores[alpha].append(score)

        # Compute mean score for each alpha and find the best alpha
        mean_scores = {alpha: np.mean(scores) for alpha, scores in alpha_scores.items()}
        self.best_alpha_ = max(mean_scores, key=mean_scores.get)
        self.best_score_ = mean_scores[self.best_alpha_]

        # Refit model on the entire dataset with the best alpha
        XtX = X.T @ X
        XtY = X.T @ y
        u, Q = scipy.linalg.eigh(XtX)
        Y_prime = Q.T @ XtY
        d = 1 / (u + self.best_alpha_)
        D = np.diag(d)
        self.coef_ = Q @ D @ Y_prime

        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_responses)
            Predicted values.
        """
        return X @ self.coef_

    def score(self, X, y):
        """
        Evaluate the model using the scoring function.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix.
        y : ndarray of shape (n_samples,) or (n_samples, n_responses)
            True target values.

        Returns
        -------
        score : float
            Model performance score.
        """
        y_pred = self.predict(X)
        return self.scoring(y, y_pred)
