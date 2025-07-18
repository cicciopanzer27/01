#!/usr/bin/env python3
"""
Command-line interface for M.I.A.-simbolic benchmarks.

This module provides a command-line interface for running benchmarks
and comparing the performance of the M.I.A.-simbolic optimizer with
other optimization algorithms.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mia_simbolic import MIAOptimizer, BenchmarkSuite, print_banner
from mia_simbolic.benchmarks.problems import (
    SphereFunction, RosenbrockFunction, RastriginFunction, 
    AckleyFunction, NeuralNetworkLoss, PortfolioOptimization
)
from mia_simbolic.benchmarks.baselines import (
    AdamOptimizer, SGDOptimizer, LBFGSOptimizer, 
    RMSpropOptimizer, AdagradOptimizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_benchmark(args):
    """Run benchmark command."""
    print_banner()
    print("\nRunning M.I.A.-simbolic Benchmarks\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create benchmark suite
    suite = BenchmarkSuite(
        n_runs=args.runs,
        output_dir=str(output_dir)
    )
    
    # Add baseline optimizers
    if args.baselines:
        suite.add_optimizer('Adam', lambda: AdamOptimizer(lr=0.01))
        suite.add_optimizer('SGD+Momentum', lambda: SGDOptimizer(lr=0.01, momentum=0.9))
        suite.add_optimizer('L-BFGS', lambda: LBFGSOptimizer())
        suite.add_optimizer('RMSprop', lambda: RMSpropOptimizer(lr=0.01))
        suite.add_optimizer('Adagrad', lambda: AdagradOptimizer(lr=0.01))
    
    # Run benchmarks
    dimensions = [int(d) for d in args.dimensions.split(',')]
    
    print(f"Running benchmarks with dimensions: {dimensions}")
    print(f"Number of runs per problem: {args.runs}")
    print(f"Output directory: {output_dir}")
    print(f"Including baselines: {args.baselines}")
    print(f"Problem class: {args.problem_class}")
    print("\nThis may take a while...\n")
    
    start_time = time.time()
    
    results = suite.run(
        problem_class=args.problem_class,
        dimensions=dimensions,
        optimizers=None  # Use all available optimizers
    )
    
    end_time = time.time()
    
    # Generate comparison
    comparison = suite.compare(results)
    
    # Plot results
    suite.plot_results(results, filename="benchmark_results.png")
    
    print(f"\nBenchmarks completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    
    # Print summary
    summary = results.summary()
    
    print("\nSummary of Results:")
    print("=" * 80)
    
    for key, stats in summary.items():
        print(f"\n{key}:")
        print(f"  Problem: {stats['problem']}")
        print(f"  Dimension: {stats['dimension']}")
        print(f"  Convergence rate: {stats['convergence_rate']:.2f}")
        print(f"  Mean time: {stats['mean_time']:.4f} seconds")
        print(f"  Mean iterations: {stats['mean_iterations']:.1f}")
        print(f"  Mean final value: {stats['mean_final_value']:.6e}")
    
    print("\nComparison of Optimizers:")
    print("=" * 80)
    
    for problem_key, metrics in comparison.items():
        print(f"\n{problem_key}:")
        
        # Print best optimizer for each metric
        print(f"  Best convergence rate: {metrics.get('best_convergence_rate', 'N/A')}")
        print(f"  Best mean time: {metrics.get('best_mean_time', 'N/A')}")
        print(f"  Best mean iterations: {metrics.get('best_mean_iterations', 'N/A')}")
        print(f"  Best mean final value: {metrics.get('best_mean_final_value', 'N/A')}")
    
    print(f"\nDetailed results and plots saved to: {output_dir}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="M.I.A.-simbolic Benchmarks")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("run", help="Run benchmarks")
    benchmark_parser.add_argument("--dimensions", type=str, default="10,50,100",
                                help="Comma-separated list of dimensions to test")
    benchmark_parser.add_argument("--runs", type=int, default=5,
                                help="Number of runs per problem")
    benchmark_parser.add_argument("--output-dir", type=str, default="results/benchmarks",
                                help="Output directory for results")
    benchmark_parser.add_argument("--baselines", action="store_true",
                                help="Include baseline optimizers")
    benchmark_parser.add_argument("--problem-class", type=str, default="all",
                                help="Problem class to benchmark (all, sphere, rosenbrock, etc.)")
    benchmark_parser.add_argument("--verbose", action="store_true",
                                help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == "run":
        run_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()