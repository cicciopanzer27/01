#!/usr/bin/env python3
"""
Verify Implementation Script

This script verifies that the M.I.A.-simbolic implementation works correctly
by running a simple optimization problem and checking the results.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.mia_simbolic import MIAOptimizer, print_banner
from src.mia_simbolic.utils.monitoring import OptimizationMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_simple_test():
    """Run a simple test to verify the implementation."""
    print_banner()
    print("\nVerifying M.I.A.-simbolic Implementation\n")
    
    # Create output directory
    output_dir = Path("results/verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create monitor
    monitor = OptimizationMonitor(
        metrics=['loss', 'gradient_norm', 'step_size'],
        update_frequency=10,
        save_plots=True,
        output_dir=str(output_dir)
    )
    
    # Create optimizer
    optimizer = MIAOptimizer(
        convergence_tolerance=1e-6,
        max_iterations=100,
        auto_tune=False,
        monitor=monitor
    )
    
    # Define test functions
    test_functions = {
        "Sphere": lambda x: np.sum(x**2),
        "Rosenbrock": lambda x: sum(100.0 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1))
    }
    
    # Run tests
    results = {}
    
    for name, func in test_functions.items():
        print(f"\nTesting {name} function...")
        
        # Initial point
        initial_point = np.ones(10) if name == "Sphere" else np.zeros(10)
        
        # Run optimization
        start_time = time.time()
        result = optimizer.optimize(func, initial_point)
        end_time = time.time()
        
        # Store results
        results[name] = {
            "converged": result.converged,
            "iterations": result.nit,
            "final_value": result.fun,
            "time": end_time - start_time,
            "gradient_norm": result.gradient_norm
        }
        
        # Print results
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.nit}")
        print(f"  Final value: {result.fun:.6e}")
        print(f"  Time: {end_time - start_time:.4f} seconds")
        print(f"  Gradient norm: {result.gradient_norm:.6e}")
    
    # Save monitor results
    monitor.save_history()
    monitor.generate_report()
    
    # Verify results
    all_passed = True
    
    for name, result in results.items():
        if not result["converged"]:
            print(f"\n❌ {name} test failed: Did not converge")
            all_passed = False
        elif result["final_value"] > 1e-4:
            print(f"\n❌ {name} test failed: Final value too high ({result['final_value']:.6e})")
            all_passed = False
        else:
            print(f"\n✅ {name} test passed!")
    
    if all_passed:
        print("\n🎉 All tests passed! M.I.A.-simbolic implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    print(f"\nResults and plots saved to: {output_dir}")
    
    return all_passed

if __name__ == "__main__":
    success = run_simple_test()
    sys.exit(0 if success else 1)