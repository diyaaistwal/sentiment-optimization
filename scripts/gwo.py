import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def gwo_optimizer(func, bounds, n_wolves=30, max_iter=100):
    dim = len(bounds)
    wolves = np.random.rand(n_wolves, dim)
    alpha = np.zeros(dim)
    beta = np.zeros(dim)
    delta = np.zeros(dim)

    alpha_score = float('inf')
    beta_score = float('inf')
    delta_score = float('inf')

    # GWO parameters
    a = 2  # Convergence factor

    logging.info(f"GWO optimization started with {n_wolves} wolves and {max_iter} iterations.")

    for iteration in range(max_iter):
        logging.info(f"Iteration {iteration + 1}/{max_iter}")

        for i in range(n_wolves):
            fitness = func(wolves[i])

            # Update Alpha, Beta, Delta wolves
            if fitness < alpha_score:
                alpha_score = fitness
                alpha = wolves[i]
                logging.debug(f"Alpha wolf updated to position {alpha} with score {alpha_score}.")
            elif fitness < beta_score:
                beta_score = fitness
                beta = wolves[i]
                logging.debug(f"Beta wolf updated to position {beta} with score {beta_score}.")
            elif fitness < delta_score:
                delta_score = fitness
                delta = wolves[i]
                logging.debug(f"Delta wolf updated to position {delta} with score {delta_score}.")

        # Update the positions of the wolves
        a -= 2 / max_iter
        logging.debug(f"Convergence factor 'a' updated to {a:.4f}.")

        for i in range(n_wolves):
            r1, r2 = np.random.rand(2)
            A = 2 * a * r1 - a
            C = 2 * r2
            D_alpha = np.abs(C * alpha - wolves[i])
            D_beta = np.abs(C * beta - wolves[i])
            D_delta = np.abs(C * delta - wolves[i])

            wolves[i] = wolves[i] + A * (D_alpha + D_beta + D_delta) / 3

        # Ensure wolves stay within bounds
        wolves = np.clip(wolves, bounds[:, 0], bounds[:, 1])

        logging.debug(f"Updated wolves' positions: {wolves}")

    logging.info(f"GWO optimization completed. Best position: {alpha}, Best score: {alpha_score}")
    return alpha, alpha_score


if __name__ == "__main__":
    def example_function(x):
        return np.sum(x**2)

    bounds = np.array([[-5, 5], [-5, 5]])  # Example bounds for 2D optimization
    best_position, best_score = gwo_optimizer(example_function, bounds)
    logging.info(f"Best position: {best_position}, Best score: {best_score}")
