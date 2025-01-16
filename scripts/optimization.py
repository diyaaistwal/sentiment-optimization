import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from pyswarm import pso  # Import PSO
from gwo import gwo_optimizer
from joblib import Parallel, delayed  # Importing parallelization


# Function to optimize SVM hyperparameters using GridSearchCV
def optimize_svm_hyperparameters(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],  # Reduced range for faster optimization
        'gamma': [0.01, 0.1]  # Reduced range for faster optimization
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1)  # Reduced folds and parallelized
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

# Function to optimize SVM hyperparameters using PSO
def optimize_svm_hyperparameters_pso(X_train, y_train, max_iter=10):
    """
    Optimize SVM hyperparameters using Particle Swarm Optimization (PSO)

    :param X_train: Features for training
    :param y_train: Target labels for training
    :param max_iter: Maximum number of iterations for optimization
    :return: Best hyperparameters found during optimization
    """
    def svm_objective(params):
        C, gamma = params
        model = SVC(C=C, gamma=gamma)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        return 1 - accuracy_score(y_train, y_pred)

    bounds = np.array([[0.1, 10], [0.01, 0.1]])  # Smaller bounds for faster optimization
    
    # Run PSO optimization in parallel (use 'maxiter' to limit iterations)
    best_params, _ = pso(svm_objective, bounds[:, 0], bounds[:, 1], maxiter=max_iter, swarmsize=10)
    return {'C': best_params[0], 'gamma': best_params[1]}

# Function to optimize SVM hyperparameters using GWO
def optimize_svm_hyperparameters_gwo(X_train, y_train, max_iter=10):
    """
    Optimize SVM hyperparameters using Grey Wolf Optimization (GWO)

    :param X_train: Features for training
    :param y_train: Target labels for training
    :param max_iter: Maximum number of iterations for optimization
    :return: Best hyperparameters found during optimization
    """
    def svm_objective(params):
        C, gamma = params
        model = SVC(C=C, gamma=gamma)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        return 1 - accuracy_score(y_train, y_pred)

    bounds = np.array([[0.1, 10], [0.01, 0.1]])  # Smaller bounds for faster optimization
    
    # Run GWO optimization in parallel (use 'maxiter' to limit iterations)
    best_params, _ = gwo_optimizer(svm_objective, bounds, max_iter=max_iter)
    return {'C': best_params[0], 'gamma': best_params[1]}

# Function to train the SVM model with optimized hyperparameters
def train_optimized_model(params, X_train, y_train):
    model = SVC(C=params['C'], gamma=params['gamma'])
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test data: {accuracy * 100:.2f}%")
