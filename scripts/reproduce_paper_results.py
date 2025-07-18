#!/usr/bin/env python3
"""
Reproduce Paper Results Script

This script reproduces all the experimental results reported in the 
M.I.A.-simbolic paper. It runs the complete benchmark suite and 
generates comparison tables and figures.

Usage:
    python scripts/reproduce_paper_results.py [--quick] [--output-dir results/]
"""

import argparse
import json
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
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mia_simbolic import MIAOptimizer, BenchmarkSuite, ValidationProtocol
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

class PaperResultsReproducer:
    """Reproduces all experimental results from the paper."""
    
    def __init__(self, output_dir: str = "results/paper_reproduction", quick_mode: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_mode = quick_mode
        
        # Experimental parameters
        self.n_runs = 5 if quick_mode else 30  # Paper uses 30 runs
        self.max_iterations = 100 if quick_mode else 1000
        self.dimensions = [10, 50] if quick_mode else [10, 50, 100, 500, 1000]
        
        # Initialize optimizers
        self.optimizers = {
            'MIA-simbolic': MIAOptimizer(
                convergence_tolerance=1e-6,
                max_iterations=self.max_iterations,
                auto_tune=True
            ),
            'Adam': AdamOptimizer(lr=0.01),
            'SGD+Momentum': SGDOptimizer(lr=0.01, momentum=0.9),
            'RMSprop': RMSpropOptimizer(lr=0.01),
            'L-BFGS': LBFGSOptimizer(),
            'Adagrad': AdagradOptimizer(lr=0.01)
        }
        
        # Initialize problems
        self.problems = {
            'Sphere': SphereFunction,
            'Rosenbrock': RosenbrockFunction,
            'Rastrigin': RastriginFunction,
            'Ackley': AckleyFunction,
            'Neural Network': NeuralNetworkLoss,
            'Portfolio': PortfolioOptimization
        }
        
        self.results = {}
        
    def run_single_experiment(self, optimizer_name: str, problem_name: str, 
                            dimension: int, run_id: int) -> Dict[str, Any]:
        """Run a single optimization experiment."""
        
        logger.info(f"Running {optimizer_name} on {problem_name} (dim={dimension}, run={run_id})")
        
        # Initialize problem
        problem_class = self.problems[problem_name]
        problem = problem_class(dimension=dimension)
        
        # Initialize optimizer
        optimizer = self.optimizers[optimizer_name]
        
        # Generate random starting point
        np.random.seed(run_id)  # For reproducibility
        initial_point = np.random.randn(dimension)
        
        # Run optimization
        start_time = time.time()
        try:
            result = optimizer.optimize(problem.objective, initial_point)
            end_time = time.time()
            
            return {
                'optimizer': optimizer_name,
                'problem': problem_name,
                'dimension': dimension,
                'run_id': run_id,
                'converged': result.converged,
                'final_value': result.fun,
                'iterations': result.nit,
                'time': end_time - start_time,
                'success': result.success,
                'gradient_norm': getattr(result, 'gradient_norm', None),
                'efficiency_score': getattr(result, 'efficiency_score', None)
            }
            
        except Exception as e:
            logger.error(f"Error in {optimizer_name} on {problem_name}: {e}")
            return {
                'optimizer': optimizer_name,
                'problem': problem_name,
                'dimension': dimension,
                'run_id': run_id,
                'converged': False,
                'final_value': np.inf,
                'iterations': self.max_iterations,
                'time': np.inf,
                'success': False,
                'gradient_norm': np.inf,
                'efficiency_score': 0.0,
                'error': str(e)
            }
    
    def run_convergence_analysis(self) -> pd.DataFrame:
        """Run convergence analysis experiments (Table 1 in paper)."""
        
        logger.info("Running convergence analysis experiments...")
        
        results = []
        total_experiments = len(self.optimizers) * len(self.problems) * len(self.dimensions) * self.n_runs
        experiment_count = 0
        
        for optimizer_name in self.optimizers:
            for problem_name in self.problems:
                for dimension in self.dimensions:
                    for run_id in range(self.n_runs):
                        experiment_count += 1
                        logger.info(f"Progress: {experiment_count}/{total_experiments}")
                        
                        result = self.run_single_experiment(
                            optimizer_name, problem_name, dimension, run_id
                        )
                        results.append(result)
        
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "convergence_analysis_raw.csv", index=False)
        
        return df
    
    def run_scalability_analysis(self) -> pd.DataFrame:
        """Run scalability analysis (Figure 2 in paper)."""
        
        logger.info("Running scalability analysis...")
        
        # Focus on MIA-simbolic vs best baselines for scalability
        scalability_optimizers = ['MIA-simbolic', 'Adam', 'L-BFGS']
        scalability_dimensions = [10, 50, 100, 500, 1000, 5000] if not self.quick_mode else [10, 50, 100]
        
        results = []
        
        for optimizer_name in scalability_optimizers:
            for dimension in scalability_dimensions:
                # Use Rosenbrock as representative problem
                for run_id in range(self.n_runs):
                    result = self.run_single_experiment(
                        optimizer_name, 'Rosenbrock', dimension, run_id
                    )
                    results.append(result)
        
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "scalability_analysis_raw.csv", index=False)
        
        return df
    
    def run_robustness_analysis(self) -> pd.DataFrame:
        """Run robustness to noise analysis (Figure 3 in paper)."""
        
        logger.info("Running robustness analysis...")
        
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25] if not self.quick_mode else [0.0, 0.10, 0.20]
        results = []
        
        for noise_level in noise_levels:
            for optimizer_name in ['MIA-simbolic', 'Adam', 'L-BFGS']:
                for run_id in range(self.n_runs):
                    # Add noise to Rosenbrock function
                    problem = RosenbrockFunction(dimension=50, noise_std=noise_level)
                    
                    np.random.seed(run_id)
                    initial_point = np.random.randn(50)
                    
                    optimizer = self.optimizers[optimizer_name]
                    
                    start_time = time.time()
                    try:
                        result = optimizer.optimize(problem.objective, initial_point)
                        end_time = time.time()
                        
                        results.append({
                            'optimizer': optimizer_name,
                            'noise_level': noise_level,
                            'run_id': run_id,
                            'converged': result.converged,
                            'final_value': result.fun,
                            'time': end_time - start_time,
                            'success': result.success
                        })
                        
                    except Exception as e:
                        results.append({
                            'optimizer': optimizer_name,
                            'noise_level': noise_level,
                            'run_id': run_id,
                            'converged': False,
                            'final_value': np.inf,
                            'time': np.inf,
                            'success': False,
                            'error': str(e)
                        })
        
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "robustness_analysis_raw.csv", index=False)
        
        return df
    
    def generate_summary_tables(self, convergence_df: pd.DataFrame) -> None:
        """Generate summary tables (Tables 1-3 in paper)."""
        
        logger.info("Generating summary tables...")
        
        # Table 1: Convergence rates by optimizer and problem
        convergence_summary = convergence_df.groupby(['optimizer', 'problem']).agg({
            'converged': 'mean',
            'final_value': 'mean',
            'iterations': 'mean',
            'time': 'mean',
            'success': 'mean'
        }).round(4)
        
        convergence_summary.to_csv(self.output_dir / "table1_convergence_rates.csv")
        
        # Table 2: Performance comparison (aggregate across all problems)
        performance_summary = convergence_df.groupby('optimizer').agg({
            'converged': ['mean', 'std'],
            'time': ['mean', 'std'],
            'iterations': ['mean', 'std'],
            'final_value': ['mean', 'std']
        }).round(4)
        
        performance_summary.to_csv(self.output_dir / "table2_performance_comparison.csv")
        
        # Table 3: Statistical significance tests
        from scipy.stats import wilcoxon
        
        mia_results = convergence_df[convergence_df['optimizer'] == 'MIA-simbolic']
        significance_tests = []
        
        for optimizer in ['Adam', 'L-BFGS', 'SGD+Momentum', 'RMSprop']:
            other_results = convergence_df[convergence_df['optimizer'] == optimizer]
            
            # Wilcoxon signed-rank test for convergence times
            mia_times = mia_results['time'].values
            other_times = other_results['time'].values
            
            min_len = min(len(mia_times), len(other_times))
            if min_len > 0:
                statistic, p_value = wilcoxon(mia_times[:min_len], other_times[:min_len])
                
                significance_tests.append({
                    'comparison': f'MIA-simbolic vs {optimizer}',
                    'metric': 'time',
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        
        significance_df = pd.DataFrame(significance_tests)
        significance_df.to_csv(self.output_dir / "table3_statistical_tests.csv", index=False)
    
    def generate_figures(self, convergence_df: pd.DataFrame, 
                        scalability_df: pd.DataFrame, 
                        robustness_df: pd.DataFrame) -> None:
        """Generate figures from the paper."""
        
        logger.info("Generating figures...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Figure 1: Convergence comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        problems = ['Sphere', 'Rosenbrock', 'Rastrigin', 'Ackley', 'Neural Network', 'Portfolio']
        
        for i, problem in enumerate(problems):
            problem_data = convergence_df[convergence_df['problem'] == problem]
            
            convergence_rates = problem_data.groupby('optimizer')['converged'].mean()
            convergence_rates.plot(kind='bar', ax=axes[i], title=f'{problem} Function')
            axes[i].set_ylabel('Convergence Rate')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure1_convergence_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Scalability analysis
        plt.figure(figsize=(10, 6))
        
        for optimizer in ['MIA-simbolic', 'Adam', 'L-BFGS']:
            optimizer_data = scalability_df[scalability_df['optimizer'] == optimizer]
            scaling_data = optimizer_data.groupby('dimension')['time'].mean()
            
            plt.loglog(scaling_data.index, scaling_data.values, 'o-', label=optimizer, linewidth=2)
        
        plt.xlabel('Problem Dimension')
        plt.ylabel('Average Time (seconds)')
        plt.title('Scalability Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "figure2_scalability.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Robustness to noise
        plt.figure(figsize=(10, 6))
        
        for optimizer in ['MIA-simbolic', 'Adam', 'L-BFGS']:
            optimizer_data = robustness_df[robustness_df['optimizer'] == optimizer]
            robustness_data = optimizer_data.groupby('noise_level')['converged'].mean()
            
            plt.plot(robustness_data.index, robustness_data.values, 'o-', label=optimizer, linewidth=2)
        
        plt.xlabel('Noise Level (σ)')
        plt.ylabel('Convergence Rate')
        plt.title('Robustness to Noise')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "figure3_robustness.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Figures saved to {self.output_dir}")
    
    def generate_reproduction_report(self) -> None:
        """Generate a comprehensive reproduction report."""
        
        logger.info("Generating reproduction report...")
        
        report_content = f"""
# M.I.A.-simbolic Paper Results Reproduction Report

**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Mode:** {'Quick' if self.quick_mode else 'Full'}
**Number of runs per experiment:** {self.n_runs}
**Dimensions tested:** {self.dimensions}

## Summary

This report contains the reproduction of all experimental results from the paper:
"M.I.A.-simbolic: A Multi-Agent Approach to Threshold Optimization in Machine Learning"

## Files Generated

### Raw Data
- `convergence_analysis_raw.csv`: Raw results for convergence analysis
- `scalability_analysis_raw.csv`: Raw results for scalability analysis  
- `robustness_analysis_raw.csv`: Raw results for robustness analysis

### Summary Tables
- `table1_convergence_rates.csv`: Convergence rates by optimizer and problem
- `table2_performance_comparison.csv`: Overall performance comparison
- `table3_statistical_tests.csv`: Statistical significance tests

### Figures
- `figure1_convergence_comparison.png`: Convergence comparison across problems
- `figure2_scalability.png`: Scalability analysis
- `figure3_robustness.png`: Robustness to noise analysis

## Validation

To validate these results against the paper:

1. **Convergence Rates**: MIA-simbolic should achieve 100% convergence on all problems
2. **Speed Improvement**: MIA-simbolic should be 10-100x faster than baselines
3. **Scalability**: MIA-simbolic should scale better than O(n^2) methods
4. **Robustness**: MIA-simbolic should maintain >80% convergence up to σ=0.20

## Reproduction Instructions

To reproduce these results independently:

```bash
# Full reproduction (takes several hours)
python scripts/reproduce_paper_results.py

# Quick reproduction (takes ~30 minutes)
python scripts/reproduce_paper_results.py --quick

# Custom output directory
python scripts/reproduce_paper_results.py --output-dir my_results/
```

## System Information

- Python version: {sys.version}
- NumPy version: {np.__version__}
- Platform: {sys.platform}

## Notes

{'This is a quick reproduction with reduced parameters for faster execution.' if self.quick_mode else 'This is a full reproduction matching the paper parameters.'}

For questions about reproduction, please open an issue at:
https://github.com/mia-simbolic/optimization/issues
"""
        
        with open(self.output_dir / "reproduction_report.md", "w") as f:
            f.write(report_content)
    
    def run_full_reproduction(self) -> None:
        """Run the complete reproduction pipeline."""
        
        logger.info("Starting full paper results reproduction...")
        start_time = time.time()
        
        # Run experiments
        convergence_df = self.run_convergence_analysis()
        scalability_df = self.run_scalability_analysis()
        robustness_df = self.run_robustness_analysis()
        
        # Generate outputs
        self.generate_summary_tables(convergence_df)
        self.generate_figures(convergence_df, scalability_df, robustness_df)
        self.generate_reproduction_report()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"Reproduction completed in {total_time:.2f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("PAPER RESULTS REPRODUCTION COMPLETED")
        print("="*60)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Output directory: {self.output_dir}")
        print(f"Mode: {'Quick' if self.quick_mode else 'Full'}")
        print("\nGenerated files:")
        for file_path in sorted(self.output_dir.glob("*")):
            print(f"  - {file_path.name}")
        print("\nTo view results:")
        print(f"  cd {self.output_dir}")
        print("  ls -la")
        print("\nFor validation, compare with paper Tables 1-3 and Figures 1-3.")

def main():
    parser = argparse.ArgumentParser(description="Reproduce M.I.A.-simbolic paper results")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick reproduction with reduced parameters")
    parser.add_argument("--output-dir", default="results/paper_reproduction",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    reproducer = PaperResultsReproducer(
        output_dir=args.output_dir,
        quick_mode=args.quick
    )
    
    reproducer.run_full_reproduction()

if __name__ == "__main__":
    main()

