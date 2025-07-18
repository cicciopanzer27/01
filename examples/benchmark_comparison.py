"""
Benchmark comparison example using M.I.A.-simbolic.

This example demonstrates how to use the BenchmarkSuite to compare
the M.I.A.-simbolic optimizer with other optimization algorithms.
"""

import numpy as np
from src.mia_simbolic.utils.benchmarks import BenchmarkSuite
from src.mia_simbolic.config import Config


def main():
    """Run the benchmark comparison example."""
    # Create configuration
    config = Config(
        max_iterations=100,
        convergence_tolerance=1e-6,
        generator_learning_rate=0.01,
        auto_tune=False
    )
    
    # Create benchmark suite
    suite = BenchmarkSuite(
        n_runs=3,  # Number of runs per problem
        output_dir="results/benchmarks",
        **config.__dict__  # Pass config parameters to optimizer
    )
    
    print("Running benchmark suite...")
    print("This may take a while...")
    
    # Run benchmarks
    results = suite.run(
        problem_class="all",  # Use all available problems
        dimensions=[10, 50],  # Test with different dimensions
        optimizers=["mia_simbolic"]  # Use only M.I.A.-simbolic optimizer
    )
    
    # Print summary
    summary = results.summary()
    print("\nBenchmark Summary:")
    for key, stats in summary.items():
        print(f"\n{key}:")
        print(f"  Problem: {stats['problem']}")
        print(f"  Dimension: {stats['dimension']}")
        print(f"  Convergence rate: {stats['convergence_rate']:.2f}")
        print(f"  Mean time: {stats['mean_time']:.4f} seconds")
        print(f"  Mean iterations: {stats['mean_iterations']:.1f}")
        print(f"  Mean final value: {stats['mean_final_value']:.6e}")
    
    # Try to add scipy optimizers if available
    try:
        from scipy.optimize import minimize
        
        # Add scipy optimizers
        suite.add_optimizer('scipy_bfgs', lambda: minimize)
        suite.add_optimizer('scipy_nelder_mead', lambda: minimize)
        
        print("\nRunning comparison with scipy optimizers...")
        
        # Run comparison benchmarks (with smaller dimensions for speed)
        comparison_results = suite.run(
            problem_class="sphere",  # Use only sphere function for comparison
            dimensions=[10],  # Use smaller dimension for speed
            optimizers=["mia_simbolic", "scipy_bfgs", "scipy_nelder_mead"]
        )
        
        # Compare optimizers
        comparison = suite.compare(comparison_results)
        
        print("\nOptimizer Comparison:")
        for problem_key, metrics in comparison.items():
            print(f"\n{problem_key}:")
            
            # Print best optimizer for each metric
            print(f"  Best convergence rate: {metrics.get('best_convergence_rate', 'N/A')}")
            print(f"  Best mean time: {metrics.get('best_mean_time', 'N/A')}")
            print(f"  Best mean iterations: {metrics.get('best_mean_iterations', 'N/A')}")
            print(f"  Best mean final value: {metrics.get('best_mean_final_value', 'N/A')}")
            
            # Print relative performance
            print("\n  Relative Performance:")
            for key, value in metrics.items():
                if key.endswith('_relative'):
                    optimizer = key.split('_')[0]
                    metric = '_'.join(key.split('_')[1:-1])
                    print(f"    {optimizer} {metric}: {value:.2f}")
        
        # Plot results
        suite.plot_results(comparison_results, filename="optimizer_comparison.png")
        print("\nComparison plot saved to 'results/benchmarks/optimizer_comparison.png'")
        
    except ImportError:
        print("\nSciPy not available. Skipping comparison with scipy optimizers.")
    
    print("\nBenchmark results saved to 'results/benchmarks' directory.")


if __name__ == "__main__":
    main()