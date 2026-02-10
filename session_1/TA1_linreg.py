import pandas as pd
import jax
import jax.numpy as np
from jax import value_and_grad

jax.config.update("jax_enable_x64", True)  # we want 64-bit floats for better precision

class LinearRegressionModel:
    """
    Linear regression trained with plain gradient descent using JAX autodiff.

    Parameters:
    - learning_rate: float, optional (default=0.01)
    - n_iterations: int, optional (default=1000)
    - tol: float, optional (default=1e-6)
    - verbose: bool, optional (default=False)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tol: float = 1e-6,
        verbose: bool = False,
        standardize: bool = True
    ):
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.standardize = bool(standardize)

        # Parameters
        self.betas = None # more well known as "weights"
        self.intercept = None # more well known as "bias"
        self.params_ = {}
        self.params_standard_ = {}
        self.params_svd_ = {}
        self.norm_values_ = {"X": {}, "y": {}}

        # Training diagnostics
        self.converged_ = None
        self.n_iter_ = 0
        self.history_ = []  # list of losses over the training iterations

    # ------------------------- Utility functions -------------------------
    """
    Don't worry too much about those, we are using numpy arrays for speed, since math operates
    on them more efficiently than on other data structures.

    Actually, we use JAX numpy arrays because JAX is the library we are using for
    automatic differentiation.

    I really don't want you to worry if you don't understand what's going on here since this is
    just put in place for a matter of efficiency.
    """
    
    @staticmethod
    def _prepare(X, y):
        """Bit of a fancy way to convert X and y to jax np arrays."""

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        return X, y

    # ------------------------- Main functions -------------------------
    def fit(self, X, y):
        """
        Fit linear regression by gradient descent on MSE.

        Parameters
        ----------
        X : pandas.DataFrame or array-like, shape (n_samples, n_features)
        y : pandas.Series/array-like, shape (n_samples,)
        """

        X, y = self._prepare(X, y) # we prepare the inputs so they match our desired format
        n_x, n_features = X.shape
        n_y = y.shape[0]

        # We set up the values we will use to standardize the features and target
        self.norm_values_["X"]["mean"] = np.mean(X, axis=0)
        self.norm_values_["X"]["std"] = np.std(X, axis=0)
        self.norm_values_["y"]["mean"] = np.mean(y, axis=0)
        self.norm_values_["y"]["std"] = np.std(y, axis=0)

        # Aaaaand we standardize the features and target
        if self.standardize:
            X = (X - self.norm_values_["X"]["mean"]) / self.norm_values_["X"]["std"]
            y = (y - self.norm_values_["y"]["mean"]) / self.norm_values_["y"]["std"]

        # Initialize parameters (since it is a linear regression, zero initialization is fine)
        self.betas = np.zeros((n_features,), dtype=X.dtype)
        self.intercept = np.asarray(0.0, dtype=X.dtype)

        # Define loss in a format usable in jax autodiff
        def loss_mse(B, B0):
            Z = X @ B + B0
            err = Z - y
            
            return np.mean(err * err)

        # value_and_grad, differentiation over the betas and the intercept
        vjg = value_and_grad(loss_mse, argnums=(0, 1))

        # Reset all training diagnostics
        self.history_.clear()
        self.converged_ = False
        self.n_iter_ = 0

        # Prepare the state of the parameters for the optimization step
        b, b0 = self.betas, self.intercept
        lr = self.learning_rate
        tol = self.tol

        # Optimization loop
        for i in range(1, self.n_iterations + 1):
            # Compute the value of the loss function at the current parameters
            # and also compute the gradient for each parameter
            loss_val, (grad_b, grad_0) = vjg(b, b0)
            if self.verbose:
                print(f"Iteration {i}: loss (MSE) = {float(loss_val):.3e}")
            self.history_.append(float(loss_val))

            # GRADIENT DESCENT CRUCIAL STEP HERE
            # Parameter update using the learning rate (plain Gradient Descent)
            b = b - lr * grad_b
            b0 = b0 - lr * grad_0

            # Convergence check: gradient norm
            # We compute the norm of all the gradients. If we are close enough to 0,
            # the norm should be below the tolerance level that we set 
            # (in this case, tol = 0.000001 if no other tol is set)
            grad_norm = np.sqrt(np.sum(grad_b * grad_b) + grad_0 * grad_0)
            if grad_norm < tol:
                self.converged_ = True
                self.n_iter_ = i
                break
            elif i == self.n_iterations:
                # if we arrive at the last iteration without convergence, we print a warning
                self.n_iter_ = self.n_iterations
                if self.verbose:
                    print(
                        "WARNING: Gradient descent did not converge"
                        f"within {self.n_iterations} iterations (grad_norm={float(grad_norm):.3e})."
                    )

        # Establish the learned parameters
        # But since I want to compare them to the closed-form solution,
        # In this special case, I will de-standardize the betas directly
        # instead of the more standard practice of de-standardizing only the predictions
        # Spoiler alert, it gets a bit messy:
        if self.standardize:
            xbar, ybar, sx, sy = self.norm_values_["X"]["mean"], self.norm_values_["y"]["mean"], self.norm_values_["X"]["std"], self.norm_values_["y"]["std"]
            self.betas = b * sy / sx # Vector times scalar and element-wise division by another vector of same length = vector of same length
            self.intercept = b0 * sy + ybar - np.sum(self.betas * xbar) # We could also do self.betas @ xbar or any other dot product option
        else:
            self.betas = b
            self.intercept = b0
        
        self.params_.clear()
        self.params_["beta_0"] = self.intercept
        for i in range(1, len(self.betas)+1):
            self.params_[f"beta_{i}"] = self.betas[i-1]
        
        return self

    def predict(self, X):
        """Predict using the learned linear model."""
        X = self._to_jnp(X)

        return (X @ self.betas + self.intercept).reshape(-1)

    def fit_closed_form_linreg(self, X, y):
        """
        Fit a linear regression using the closed form solution that we all know and love.
        """

        X, y = self._prepare(X, y)
        
        # We need to add a bias term (intercept) into the X in this case:
        print(X)
        X = np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])
        print(X)

        # Closed-form solution: (X^T X)^(-1) X^T y
        XTX = X.T @ X
        XTX_inv = np.linalg.inv(XTX)
        Betas = XTX_inv @ (X.T @ y)

        self.params_standard_.clear()
        for i in range(len(Betas)):
            self.params_standard_[f"beta_{i}"] = Betas[i]
        
        return self

    def fit_svd_linreg(self, X, y):
        """
        Fit linear regression using SVD built from eigendecomposition of X^T X.
        Beta = V * Σ⁺ * Uᵀ * y  with  X = U Σ Vᵀ.
        """

        X, y = self._prepare(X, y)
        y = y.reshape(-1, 1)  # make y a column vector
        
        # We need to add a bias term (intercept) into the X in this case:
        X = np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])
        print("X.shape:", X.shape, "y.shape:", y.shape)

        # SVD decomposition of X. We could use the numpy linear algebra implementation of the SVD 
        # if we wanted a less involved approach.
        # U, S, VT = np.linalg.svd(X, full_matrices=False)
        
        ### HERE WE START TO COMPUTE THE CODE TO REPLACE np.linalg.svd() CALL ###
        # First we compute the covariance matrix X^T X
        XtX = X.T @ X
        print("XtX.shape:", XtX.shape)
        # Then we compute the eigen-decomposition of X^T X
        eigvals, V = np.linalg.eigh(XtX)
        print("eigvals.shape:", eigvals.shape, "V.shape:", V.shape)
        # Clip negative eigenvalues to zero
        eigvals = np.clip(eigvals, 0.0, None)
        # Take square roots of eigenvalues to get singular values
        S = np.sqrt(eigvals)
        # Sort descending singular values 
        idx = np.argsort(-S)
        S = S[idx]
        V = V[:, idx]
        # print(S.shape, V.shape)
        # Keep only big enough singular values
        tol = S[0] * 1e-12
        keep_idx = S > tol
        S = S[keep_idx]
        V = V[:, keep_idx]
        print("S.shape:", S.shape, "V.shape:", V.shape)
        # Compute U from X, V, S
        U = (X @ V) / S
        print("U.shape:", U.shape)
        VT = V.T

        ### BELOW HERE CODE CONTINUES AFTER np.linalg.svd() CALL ###
        # Build the pseudo-inverse of the diagonal matrix of singular values
        singular_values_inv = np.where(S > 1e-12, 1.0 / S, 0.0).astype(X.dtype)
        S_inv = np.diag(singular_values_inv)
        print("S_inv.shape:", S_inv.shape)

        # Moore-Penrose pseudoinverse using the SVD factors
        pseudo_inv = VT.T @ S_inv @ U.T
        print("pseudo_inv.shape:", pseudo_inv.shape)
        Betas = pseudo_inv @ y
        print("Betas.shape:", Betas.shape)

        self.params_svd_.clear()
        for i in range(len(Betas)):
            self.params_svd_[f"beta_{i}"] = Betas[i][0]

        return self

    def get_params(self):
        """Return a dict with learned parameters and diagnostics."""
        premium_params = self.params_.copy()
        premium_params["converged"] = "Converged" if self.converged_ else "Not converged"
        premium_params["n_iter"] = self.n_iter_
        premium_params["final_loss"] = None if not self.history_ else self.history_[-1]
        premium_params["closed_form_params"] = self.params_standard_
        premium_params["svd_params"] = self.params_svd_

        return premium_params
