"""
Validation protocol example using M.I.A.-simbolic.

This example demonstrates how to use the ValidationProtocol to validate
the M.I.A.-simbolic optimizer on standard test problems.
"""

import numpy as np
from src.mia_simbolic.utils.validation import ValidationProtocol
from src.mia_simbolic.config import Config


def main():
    """Run the validation protocol example."""
    # Create configuration
    config = Config(
        max_iterations=100,
        convergence_tolerance=1e-6,
        generator_learning_rate=0.01,
        auto_tune=False
    )
    
    # Create validation protocol
    protocol = ValidationProtocol(
        n_runs=3,  # Number of runs per problem
        output_dir="results/validation",
        config=config
    )
    
    print("Running validation protocol...")
    print("This may take a while...")
    
    # Run validation
    results = protocol.run(
        problem_class="all",  # Use all available problems
        dimensions=[10, 50]  # Test with different dimensions
    )
    
    # Print results
    print("\nValidation Results:")
    print(f"Overall passed: {results.overall_passed}")
    
    # Print summary
    summary = results.summary()
    for key, stats in summary.items():
        print(f"\n{key}:")
        print(f"  Problem: {stats['problem']}")
        print(f"  Dimension: {stats['dimension']}")
        print(f"  Pass rate: {stats['pass_rate']:.2f}")
        print(f"  Convergence rate: {stats['convergence_rate']:.2f}")
        print(f"  Mean time: {stats['mean_time']:.4f} seconds")
        print(f"  Mean iterations: {stats['mean_iterations']:.1f}")
        print(f"  Mean relative error: {stats['mean_relative_error']:.6e}")
        print(f"  All passed: {stats['all_passed']}")
    
    # Generate report
    protocol.generate_report(results, filename="validation_report.html")
    
    print("\nValidation report saved to 'results/validation/validation_report.html'")


if __name__ == "__main__":
    main()