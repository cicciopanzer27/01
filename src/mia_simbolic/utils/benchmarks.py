"""
Benchmarks module for M.I.A.-simbolic.

This module implements the BenchmarkSuite class, which provides
benchmarking functionality for the M.I.A.-simbolic optimizer.
"""

import logging
import time
import os
import json
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np

from ..core.optimizer import MIAOptimizer
from ..config import Config

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a benchmark run.
    
    Attributes:
        optimizer_name: Name of the optimizer
        problem_name: Name of the problem
        dimension: Dimension of the problem
        run_id: ID of the run
        converged: Whether the optimization converged
        final_value: Final function value
        iterations: Number of iterations
        time: Time taken for optimization
        success: Whether the optimization was successful
        gradient_norm: Norm of the gradient at the optimal point
        efficiency_score: Efficiency score of the optimization
    """
    optimizer_name: str
    problem_name: str
    dimension: int
    run_id: int
    converged: bool
    final_value: float
    iterations: int
    time: float
    success: bool
    gradient_norm: Optional[float] = None
    efficiency_score: Optional[float] = None
    error: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Collection of benchmark results.
    
    Attributes:
        results: List of benchmark results
        timestamp: Timestamp of the benchmark
        config: Configuration used for the benchmark
    """
    results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary.
        
        Returns:
            Dictionary representation of results
        """
        return {
            'results': [asdict(result) for result in self.results],
            'timestamp': self.timestamp,
            'config': self.config
        }
    
    def save(self, filename: str) -> None:
        """Save results to file.
        
        Args:
            filename: Name of the file to save
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved benchmark results to {filename}")
    
    @classmethod
    def load(cls, filename: str) -> 'BenchmarkResults':
        """Load results from file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Loaded benchmark results
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        results = cls(
            results=[BenchmarkResult(**result) for result in data['results']],
            timestamp=data['timestamp'],
            config=data['config']
        )
        
        logger.info(f"Loaded benchmark results from {filename}")
        
        return results
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary of results.
        
        Returns:
            Dictionary with summary statistics
        """
        # Group results by optimizer and problem
        grouped = {}
        for result in self.results:
            key = (result.optimizer_name, result.problem_name, result.dimension)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Compute statistics
        summary = {}
        for (optimizer, problem, dimension), results in grouped.items():
            key = f"{optimizer}_{problem}_{dimension}"
            summary[key] = {
                'optimizer': optimizer,
                'problem': problem,
                'dimension': dimension,
                'num_runs': len(results),
                'convergence_rate': sum(r.converged for r in results) / len(results),
                'mean_time': np.mean([r.time for r in results]),
                'std_time': np.std([r.time for r in results]),
                'mean_iterations': np.mean([r.iterations for r in results]),
                'std_iterations': np.std([r.iterations for r in results]),
                'mean_final_value': np.mean([r.final_value for r in results]),
                'std_final_value': np.std([r.final_value for r in results]),
                'mean_gradient_norm': np.mean([r.gradient_norm for r in results if r.gradient_norm is not None]),
                'mean_efficiency_score': np.mean([r.efficiency_score for r in results if r.efficiency_score is not None])
            }
        
        return summary


class BenchmarkProblem:
    """Base class for benchmark problems.
    
    Attributes:
        name: Name of the problem
        dimension: Dimension of the problem
        bounds: Bounds for variables
    """
    
    def __init__(self, name: str, dimension: int, bounds: Optional[List[Tuple[float, float]]] = None):
        """Initialize the benchmark problem.
        
        Args:
            name: Name of the problem
            dimension: Dimension of the problem
            bounds: Bounds for variables
        """
        self.name = name
        self.dimension = dimension
        self.bounds = bounds or [(-10.0, 10.0)] * dimension
        
        logger.debug(f"Initialized {self.name} problem with dimension {self.dimension}")
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function.
        
        Args:
            x: Point at which to evaluate the gradient
            
        Returns:
            Gradient vector
        """
        # Default implementation uses finite differences
        eps = 1e-6
        n = len(x)
        grad = np.zeros(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            f_plus = self.objective(x_plus)
            f_minus = self.objective(x_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    def initial_point(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate an initial point for optimization.
        
        Args:
            seed: Random seed
            
        Returns:
            Initial point
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random point within bounds
        x = np.zeros(self.dimension)
        for i in range(self.dimension):
            x[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
        
        return x


class SphereFunction(BenchmarkProblem):
    """Sphere function benchmark problem.
    
    f(x) = sum(x_i^2)
    """
    
    def __init__(self, dimension: int = 10, noise_std: float = 0.0):
        """Initialize the sphere function problem.
        
        Args:
            dimension: Dimension of the problem
            noise_std: Standard deviation of noise
        """
        super().__init__("Sphere", dimension)
        self.noise_std = noise_std
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        value = np.sum(x**2)
        
        # Add noise if specified
        if self.noise_std > 0:
            value += np.random.normal(0, self.noise_std)
        
        return value
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function.
        
        Args:
            x: Point at which to evaluate the gradient
            
        Returns:
            Gradient vector
        """
        return 2 * x


class RosenbrockFunction(BenchmarkProblem):
    """Rosenbrock function benchmark problem.
    
    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    """
    
    def __init__(self, dimension: int = 10, noise_std: float = 0.0):
        """Initialize the Rosenbrock function problem.
        
        Args:
            dimension: Dimension of the problem
            noise_std: Standard deviation of noise
        """
        super().__init__("Rosenbrock", dimension)
        self.noise_std = noise_std
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        value = 0.0
        for i in range(self.dimension - 1):
            value += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
        
        # Add noise if specified
        if self.noise_std > 0:
            value += np.random.normal(0, self.noise_std)
        
        return value


class RastriginFunction(BenchmarkProblem):
    """Rastrigin function benchmark problem.
    
    f(x) = 10 * n + sum(x_i^2 - 10 * cos(2 * pi * x_i))
    """
    
    def __init__(self, dimension: int = 10, noise_std: float = 0.0):
        """Initialize the Rastrigin function problem.
        
        Args:
            dimension: Dimension of the problem
            noise_std: Standard deviation of noise
        """
        super().__init__("Rastrigin", dimension)
        self.noise_std = noise_std
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        value = 10 * self.dimension + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        
        # Add noise if specified
        if self.noise_std > 0:
            value += np.random.normal(0, self.noise_std)
        
        return value


class AckleyFunction(BenchmarkProblem):
    """Ackley function benchmark problem.
    
    f(x) = -20 * exp(-0.2 * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(2 * pi * x_i))) + 20 + e
    """
    
    def __init__(self, dimension: int = 10, noise_std: float = 0.0):
        """Initialize the Ackley function problem.
        
        Args:
            dimension: Dimension of the problem
            noise_std: Standard deviation of noise
        """
        super().__init__("Ackley", dimension)
        self.noise_std = noise_std
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        n = float(self.dimension)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        
        value = term1 + term2 + 20 + np.e
        
        # Add noise if specified
        if self.noise_std > 0:
            value += np.random.normal(0, self.noise_std)
        
        return value


class BenchmarkSuite:
    """Benchmark suite for optimization algorithms.
    
    This class provides functionality for benchmarking optimization algorithms
    on a variety of test problems.
    
    Attributes:
        problems: Dictionary of benchmark problems
        optimizers: Dictionary of optimizers
        n_runs: Number of runs per problem
        output_dir: Directory for output files
    """
    
    def __init__(self, 
                n_runs: int = 10, 
                output_dir: Optional[str] = None, 
                **kwargs):
        """Initialize the benchmark suite.
        
        Args:
            n_runs: Number of runs per problem
            output_dir: Directory for output files
            **kwargs: Additional parameters for optimizers
        """
        self.n_runs = n_runs
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'results', 'benchmarks')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize problems
        self.problems = {
            'sphere': lambda dim: SphereFunction(dimension=dim),
            'rosenbrock': lambda dim: RosenbrockFunction(dimension=dim),
            'rastrigin': lambda dim: RastriginFunction(dimension=dim),
            'ackley': lambda dim: AckleyFunction(dimension=dim)
        }
        
        # Initialize optimizers
        self.optimizers = {
            'mia_simbolic': lambda: MIAOptimizer(**kwargs)
        }
        
        # Try to import other optimizers
        try:
            from scipy.optimize import minimize
            
            self.optimizers['scipy_bfgs'] = lambda: minimize
            self.optimizers['scipy_nelder_mead'] = lambda: minimize
            self.optimizers['scipy_powell'] = lambda: minimize
            
            logger.debug("Added scipy optimizers to benchmark suite")
        except ImportError:
            logger.warning("scipy not found, scipy optimizers will not be available")
        
        logger.debug(f"Initialized BenchmarkSuite with n_runs={n_runs}")
    
    def add_problem(self, name: str, problem_factory: Callable[[int], BenchmarkProblem]) -> None:
        """Add a problem to the benchmark suite.
        
        Args:
            name: Name of the problem
            problem_factory: Factory function to create the problem
        """
        self.problems[name] = problem_factory
        logger.debug(f"Added problem {name} to benchmark suite")
    
    def add_optimizer(self, name: str, optimizer_factory: Callable[[], Any]) -> None:
        """Add an optimizer to the benchmark suite.
        
        Args:
            name: Name of the optimizer
            optimizer_factory: Factory function to create the optimizer
        """
        self.optimizers[name] = optimizer_factory
        logger.debug(f"Added optimizer {name} to benchmark suite")
    
    def run(self, 
           problem_class: str = "all", 
           dimensions: Optional[List[int]] = None, 
           optimizers: Optional[List[str]] = None) -> BenchmarkResults:
        """Run the benchmark suite.
        
        Args:
            problem_class: Class of problems to benchmark
            dimensions: List of dimensions to benchmark
            optimizers: List of optimizers to benchmark
            
        Returns:
            Benchmark results
        """
        # Set default values
        dimensions = dimensions or [10, 50, 100]
        optimizers = optimizers or list(self.optimizers.keys())
        
        # Select problems
        if problem_class == "all":
            problems = list(self.problems.keys())
        else:
            problems = [problem_class]
        
        # Initialize results
        results = BenchmarkResults(
            config={
                'n_runs': self.n_runs,
                'problems': problems,
                'dimensions': dimensions,
                'optimizers': optimizers
            }
        )
        
        # Run benchmarks
        total_benchmarks = len(problems) * len(dimensions) * len(optimizers) * self.n_runs
        benchmark_count = 0
        
        logger.info(f"Starting benchmark suite with {total_benchmarks} benchmarks")
        start_time = time.time()
        
        for problem_name in problems:
            for dimension in dimensions:
                for optimizer_name in optimizers:
                    for run_id in range(self.n_runs):
                        benchmark_count += 1
                        logger.info(f"Running benchmark {benchmark_count}/{total_benchmarks}: "
                                   f"{optimizer_name} on {problem_name} (dim={dimension}, run={run_id})")
                        
                        # Create problem
                        problem = self.problems[problem_name](dimension)
                        
                        # Generate initial point
                        initial_point = problem.initial_point(seed=run_id)
                        
                        # Create optimizer
                        optimizer = self.optimizers[optimizer_name]()
                        
                        # Run optimization
                        try:
                            if optimizer_name.startswith('scipy_'):
                                # Handle scipy optimizers
                                method = optimizer_name.split('_')[1]
                                start_time_opt = time.time()
                                scipy_result = optimizer(
                                    problem.objective,
                                    initial_point,
                                    method=method,
                                    bounds=problem.bounds
                                )
                                end_time_opt = time.time()
                                
                                # Convert scipy result to our format
                                benchmark_result = BenchmarkResult(
                                    optimizer_name=optimizer_name,
                                    problem_name=problem_name,
                                    dimension=dimension,
                                    run_id=run_id,
                                    converged=scipy_result.success,
                                    final_value=scipy_result.fun,
                                    iterations=scipy_result.nit,
                                    time=end_time_opt - start_time_opt,
                                    success=scipy_result.success,
                                    gradient_norm=np.linalg.norm(problem.gradient(scipy_result.x)) if hasattr(problem, 'gradient') else None
                                )
                            else:
                                # Handle our optimizers
                                start_time_opt = time.time()
                                result = optimizer.optimize(
                                    problem.objective,
                                    initial_point,
                                    bounds=problem.bounds
                                )
                                end_time_opt = time.time()
                                
                                benchmark_result = BenchmarkResult(
                                    optimizer_name=optimizer_name,
                                    problem_name=problem_name,
                                    dimension=dimension,
                                    run_id=run_id,
                                    converged=result.converged,
                                    final_value=result.fun,
                                    iterations=result.nit,
                                    time=end_time_opt - start_time_opt,
                                    success=result.success,
                                    gradient_norm=result.gradient_norm,
                                    efficiency_score=result.efficiency_score
                                )
                            
                            results.results.append(benchmark_result)
                        
                        except Exception as e:
                            logger.error(f"Error in {optimizer_name} on {problem_name}: {e}")
                            
                            # Record error
                            benchmark_result = BenchmarkResult(
                                optimizer_name=optimizer_name,
                                problem_name=problem_name,
                                dimension=dimension,
                                run_id=run_id,
                                converged=False,
                                final_value=float('inf'),
                                iterations=0,
                                time=0.0,
                                success=False,
                                error=str(e)
                            )
                            
                            results.results.append(benchmark_result)
        
        end_time = time.time()
        logger.info(f"Benchmark suite completed in {end_time - start_time:.2f} seconds")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        results.save(filename)
        
        return results
    
    def compare(self, results: BenchmarkResults) -> Dict[str, Any]:
        """Compare benchmark results.
        
        Args:
            results: Benchmark results
            
        Returns:
            Dictionary with comparison results
        """
        # Get summary statistics
        summary = results.summary()
        
        # Group by problem and dimension
        grouped = {}
        for key, stats in summary.items():
            problem_key = f"{stats['problem']}_{stats['dimension']}"
            if problem_key not in grouped:
                grouped[problem_key] = {}
            grouped[problem_key][stats['optimizer']] = stats
        
        # Compare optimizers
        comparison = {}
        for problem_key, optimizers in grouped.items():
            comparison[problem_key] = {}
            
            # Find best optimizer for each metric
            for metric in ['convergence_rate', 'mean_time', 'mean_iterations', 'mean_final_value']:
                if metric == 'convergence_rate':
                    # Higher is better
                    best_optimizer = max(optimizers.items(), key=lambda x: x[1][metric])[0]
                else:
                    # Lower is better
                    best_optimizer = min(optimizers.items(), key=lambda x: x[1][metric])[0]
                
                comparison[problem_key][f"best_{metric}"] = best_optimizer
                
                # Compute relative performance
                best_value = optimizers[best_optimizer][metric]
                for optimizer, stats in optimizers.items():
                    if metric == 'convergence_rate':
                        # Higher is better
                        relative = stats[metric] / best_value if best_value > 0 else 0.0
                    else:
                        # Lower is better
                        relative = best_value / stats[metric] if stats[metric] > 0 else 0.0
                    
                    comparison[problem_key][f"{optimizer}_{metric}_relative"] = relative
        
        return comparison
    
    def plot_results(self, results: BenchmarkResults, filename: Optional[str] = None) -> None:
        """Plot benchmark results.
        
        Args:
            results: Benchmark results
            filename: Name of the file to save
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get summary statistics
            summary = results.summary()
            
            # Group by problem and dimension
            grouped = {}
            for key, stats in summary.items():
                problem_key = f"{stats['problem']}_{stats['dimension']}"
                if problem_key not in grouped:
                    grouped[problem_key] = {}
                grouped[problem_key][stats['optimizer']] = stats
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot convergence rate
            ax = axes[0, 0]
            for problem_key, optimizers in grouped.items():
                x = list(range(len(optimizers)))
                y = [stats['convergence_rate'] for stats in optimizers.values()]
                ax.bar(x, y, label=problem_key)
                ax.set_xticks(x)
                ax.set_xticklabels(list(optimizers.keys()), rotation=45)
            
            ax.set_xlabel('Optimizer')
            ax.set_ylabel('Convergence Rate')
            ax.set_title('Convergence Rate by Optimizer and Problem')
            ax.legend()
            
            # Plot mean time
            ax = axes[0, 1]
            for problem_key, optimizers in grouped.items():
                x = list(range(len(optimizers)))
                y = [stats['mean_time'] for stats in optimizers.values()]
                ax.bar(x, y, label=problem_key)
                ax.set_xticks(x)
                ax.set_xticklabels(list(optimizers.keys()), rotation=45)
            
            ax.set_xlabel('Optimizer')
            ax.set_ylabel('Mean Time (s)')
            ax.set_title('Mean Time by Optimizer and Problem')
            ax.legend()
            
            # Plot mean iterations
            ax = axes[1, 0]
            for problem_key, optimizers in grouped.items():
                x = list(range(len(optimizers)))
                y = [stats['mean_iterations'] for stats in optimizers.values()]
                ax.bar(x, y, label=problem_key)
                ax.set_xticks(x)
                ax.set_xticklabels(list(optimizers.keys()), rotation=45)
            
            ax.set_xlabel('Optimizer')
            ax.set_ylabel('Mean Iterations')
            ax.set_title('Mean Iterations by Optimizer and Problem')
            ax.legend()
            
            # Plot mean final value
            ax = axes[1, 1]
            for problem_key, optimizers in grouped.items():
                x = list(range(len(optimizers)))
                y = [stats['mean_final_value'] for stats in optimizers.values()]
                ax.bar(x, y, label=problem_key)
                ax.set_xticks(x)
                ax.set_xticklabels(list(optimizers.keys()), rotation=45)
            
            ax.set_xlabel('Optimizer')
            ax.set_ylabel('Mean Final Value')
            ax.set_title('Mean Final Value by Optimizer and Problem')
            ax.legend()
            
            # Save figure
            plt.tight_layout()
            
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(self.output_dir, f"benchmark_plot_{timestamp}.png")
            else:
                filename = os.path.join(self.output_dir, filename)
            
            plt.savefig(filename)
            plt.close(fig)
            
            logger.info(f"Saved benchmark plot to {filename}")
        
        except ImportError:
            logger.warning("matplotlib not found, plotting is not available")
        except Exception as e:
            logger.error(f"Error plotting results: {e}")