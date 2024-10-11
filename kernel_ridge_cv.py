from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

class KernelRidgeCV(BaseEstimator, RegressorMixin):
    def __init__(self, alphas=[1e-2, 1e-1, 1, 1e1, 1e2], gammas=[1e-2, 1e-1, 1, 1e1, 1e2], cv=2, scoring=None):
        """
        Custom Kernel Ridge Regression with cross-validated selection of alpha and gamma
        for the RBF kernel.

        Parameters:
        alphas (list): List of regularization strength (alpha) values to try.
        gammas (list): List of gamma values to try for the RBF kernel.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring function to use for evaluation. Defaults to R^2 for regression.
        """
        self.alphas = alphas
        self.gammas = gammas
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_estimator_ = None

    def fit(self, X, y):
        """
        Fit the model using cross-validation to find the best hyperparameters (alpha and gamma).
        """
        # Parameter grid with alpha and gamma values for the RBF kernel
        param_grid = {
            'alpha': self.alphas,
            'gamma': self.gammas
        }
        
        # Kernel Ridge Regression model with RBF kernel
        kr_model = KernelRidge(kernel='rbf')
        
        # GridSearchCV to find the best alpha and gamma
        grid_search = GridSearchCV(kr_model, param_grid, cv=self.cv, scoring=self.scoring)
        grid_search.fit(X, y)
        
        # Store the best parameters and the final estimator
        self.best_params_ = grid_search.best_params_
        self.best_estimator_ = grid_search.best_estimator_
        
        return self

    def predict(self, X):
        """
        Make predictions using the best estimator found by GridSearchCV.
        """
        if self.best_estimator_ is None:
            raise ValueError("You need to fit the model before making predictions.")
        return self.best_estimator_.predict(X)

    def get_best_params(self):
        """
        Return the best alpha and gamma found during grid search.
        """
        if self.best_params_ is None:
            raise ValueError("The model has not been fit yet.")
        return self.best_params_

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error
    
    # Step 1: Create a synthetic regression dataset
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
    # Step 2: Initialize KernelRidgeCV with RBF kernel and a range of alpha and gamma values
    kr_cv = KernelRidgeCV(alphas=[0.01, 0.1, 1, 10], gammas=[0.01, 0.1, 1, 10], cv=5, scoring='neg_mean_squared_error')
    
    # Step 3: Fit the model (find the best alpha and gamma)
    kr_cv.fit(X, y)
    
    # Step 4: Print the best alpha and gamma values
    print("Best hyperparameters:", kr_cv.get_best_params())
    
    # Step 5: Make predictions using the best estimator
    y_pred = kr_cv.predict(X)
    
    # Step 6: Evaluate the model's performance (e.g., MSE)
    print("Mean Squared Error:", mean_squared_error(y, y_pred))
