"""
Multi-objective optimization example using M.I.A.-simbolic.

This example demonstrates how to use the MIAOptimizer with a
custom multi-objective problem.
"""

import numpy as np
from src.mia_simbolic.core.optimizer import MIAOptimizer
from src.mia_simbolic.core.multi_objective import MultiObjectiveProblem
from src.mia_simbolic.config import Config
from src.mia_simbolic.utils.monitoring import OptimizationMonitor


class CustomMultiObjectiveProblem(MultiObjectiveProblem):
    """Custom multi-objective problem for demonstration.
    
    This problem combines three objectives:
    1. Validation loss: Sum of squares
    2. Computational cost: Sum of absolute values
    3. Regularization: L1 norm
    """
    
    def validation_loss(self, x):
        """Compute validation loss as sum of squares."""
        return np.sum(x**2)
    
    def computational_cost(self, x):
        """Compute computational cost as sum of absolute values."""
        return np.sum(np.abs(x))
    
    def regularization(self, x):
        """Compute regularization as L1 norm."""
        return np.sum(np.abs(x))


def main():
    """Run the multi-objective optimization example."""
    # Create configuration
    config = Config(
        max_iterations=200,
        convergence_tolerance=1e-6,
        generator_learning_rate=0.01,
        auto_tune=False,
        alpha=0.6,  # Weight for validation loss
        beta=0.3,   # Weight for computational cost
        gamma=0.1   # Weight for regularization
    )
    
    # Create optimizer
    optimizer = MIAOptimizer(config=config)
    
    # Create monitor
    monitor = OptimizationMonitor(
        metrics=['loss', 'gradient_norm', 'step_size'],
        update_frequency=10,
        save_plots=True,
        output_dir="results"
    )
    
    # Set monitor in optimizer
    optimizer.monitor = monitor
    
    # Create problem
    problem = CustomMultiObjectiveProblem(config)
    
    # Initial point
    initial_point = np.ones(5)
    
    print("Starting multi-objective optimization...")
    print(f"Weights: alpha={problem.alpha}, beta={problem.beta}, gamma={problem.gamma}")
    
    # Run optimization
    result = optimizer.optimize(problem.objective, initial_point)
    
    # Print results
    print("\nOptimization Results:")
    print(f"Success: {result.success}")
    print(f"Converged: {result.converged}")
    print(f"Final value: {result.fun:.6e}")
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Time: {result.time:.6f} seconds")
    print(f"Gradient norm: {result.gradient_norm:.6e}")
    print(f"Efficiency score: {result.efficiency_score:.6f}")
    print(f"Optimal point: {result.x}")
    
    # Compute individual components
    val_loss = problem.validation_loss(result.x)
    comp_cost = problem.computational_cost(result.x)
    reg = problem.regularization(result.x)
    
    print("\nIndividual Components:")
    print(f"Validation loss: {val_loss:.6e}")
    print(f"Computational cost: {comp_cost:.6e}")
    print(f"Regularization: {reg:.6e}")
    
    # Try different weights
    print("\nTrying different weights...")
    problem.update_weights(alpha=0.8, beta=0.1, gamma=0.1)
    print(f"New weights: alpha={problem.alpha}, beta={problem.beta}, gamma={problem.gamma}")
    
    # Run optimization with new weights
    result2 = optimizer.optimize(problem.objective, initial_point)
    
    # Print results
    print("\nOptimization Results with New Weights:")
    print(f"Final value: {result2.fun:.6e}")
    print(f"Optimal point: {result2.x}")
    
    # Compute individual components
    val_loss2 = problem.validation_loss(result2.x)
    comp_cost2 = problem.computational_cost(result2.x)
    reg2 = problem.regularization(result2.x)
    
    print("\nIndividual Components with New Weights:")
    print(f"Validation loss: {val_loss2:.6e}")
    print(f"Computational cost: {comp_cost2:.6e}")
    print(f"Regularization: {reg2:.6e}")
    
    # Save monitor results
    monitor.save_history()
    monitor.generate_report()
    
    print("\nResults saved to 'results' directory.")


if __name__ == "__main__":
    main()