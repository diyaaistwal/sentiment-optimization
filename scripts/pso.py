import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PSO:
    def __init__(self, objective_function, bounds, n_particles=30, max_iter=100):
        self.obj_fun = objective_function
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.dim = len(bounds)
        self.particles = np.random.rand(self.n_particles, self.dim) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        self.velocities = np.zeros_like(self.particles)
        self.best_positions = self.particles.copy()
        self.best_scores = np.array([float('inf')] * self.n_particles)

        self.global_best_position = self.particles[np.argmin(self.best_scores)]
        self.global_best_score = float('inf')

        self.w = 0.5  # Inertia
        self.c1 = 0.8  # Cognitive coefficient
        self.c2 = 0.8  # Social coefficient

        logging.info(f"PSO initialized with {self.n_particles} particles and {self.max_iter} iterations.")

    def optimize(self):
        logging.info("Starting optimization...")
        for iteration in range(self.max_iter):
            logging.info(f"Iteration {iteration + 1}/{self.max_iter}")
            for i in range(self.n_particles):
                fitness = self.obj_fun(self.particles[i])

                # Update personal best
                if fitness < self.best_scores[i]:
                    self.best_scores[i] = fitness
                    self.best_positions[i] = self.particles[i]
                    logging.debug(f"Particle {i} updated personal best to {self.best_positions[i]} with score {self.best_scores[i]}.")

                # Update global best
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.particles[i]
                    logging.info(f"Global best updated to position {self.global_best_position} with score {self.global_best_score}.")

            # Update velocities and positions
            for i in range(self.n_particles):
                self.velocities[i] = self.w * self.velocities[i] + self.c1 * np.random.rand() * (self.best_positions[i] - self.particles[i]) + self.c2 * np.random.rand() * (self.global_best_position - self.particles[i])
                self.particles[i] = self.particles[i] + self.velocities[i]

            # Ensure particles stay within bounds
            self.particles = np.clip(self.particles, self.bounds[:, 0], self.bounds[:, 1])

            logging.debug(f"Updated positions: {self.particles}")
        
        logging.info(f"Optimization completed. Global best position: {self.global_best_position} with score: {self.global_best_score}")
        return self.global_best_position, self.global_best_score

    def run(self):
        return self.optimize()


if __name__ == "__main__":
    def example_function(x):
        return np.sum(x**2)

    bounds = np.array([[-5, 5], [-5, 5]])  # Example bounds for 2D optimization
    pso = PSO(example_function, bounds)
    best_position, best_score = pso.run()
    logging.info(f"Best position: {best_position}, Best score: {best_score}")
