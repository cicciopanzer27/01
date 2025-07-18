"""
Simple optimization example using M.I.A.-simbolic.

This example demonstrates how to use the MIAOptimizer to optimize
a simple function.
"""

import numpy as np
from src.mia_simbolic.core.optimizer import MIAOptimizer
from src.mia_simbolic.config import Config
from src.mia_simbolic.utils.monitoring import OptimizationMonitor


def main():
    """Run the simple optimization example."""
    # Define a simple function to optimize (Rosenbrock function)
    def rosenbrock(x):
        return sum(100.0 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1))
    
    # Create configuration
    config = Config(
        max_iterations=200,
        convergence_tolerance=1e-6,
        generator_learning_rate=0.01,
        auto_tune=True,
        auto_tune_trials=10
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
    
    # Initial point
    initial_point = np.zeros(5)
    
    print("Starting optimization...")
    
    # Run optimization
    result = optimizer.optimize(rosenbrock, initial_point)
    
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
    
    # Save monitor results
    monitor.save_history()
    monitor.generate_report()
    
    print("\nResults saved to 'results' directory.")


if __name__ == "__main__":
    main()